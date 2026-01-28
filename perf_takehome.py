"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def add_instr(self, instr):
        """Add a complete instruction bundle (VLIW)"""
        self.instrs.append(instr)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized kernel: Keep data in scratch, pipeline scatter-gather with hash.
        """
        tmp1 = self.alloc_scratch("tmp1")
        
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        vone = self.alloc_scratch("vone", VLEN)
        vtwo = self.alloc_scratch("vtwo", VLEN)
        
        self.add("valu", ("vbroadcast", vone, one_const))
        self.add("valu", ("vbroadcast", vtwo, two_const))

        vhash_consts = []
        vhash_multipliers = []  # For multiply_add optimization on stages 0, 2, 4
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            c1 = self.scratch_const(val1)
            c3 = self.scratch_const(val3)
            vc1 = self.alloc_scratch(f"vhash_c1_{hi}", VLEN)
            vc3 = self.alloc_scratch(f"vhash_c3_{hi}", VLEN)
            self.add("valu", ("vbroadcast", vc1, c1))
            self.add("valu", ("vbroadcast", vc3, c3))
            vhash_consts.append((vc1, vc3))
            
            # For stages where op1="+", op2="+", op3="<<", we can use multiply_add
            # val_new = (val + const1) + (val << shift) = val * (1 + 2^shift) + const1
            if op1 == "+" and op2 == "+" and op3 == "<<":
                multiplier = 1 + (1 << val3)  # 1 + 2^shift
                mult_const = self.scratch_const(multiplier)
                vmult = self.alloc_scratch(f"vhash_mult_{hi}", VLEN)
                self.add("valu", ("vbroadcast", vmult, mult_const))
                vhash_multipliers.append((hi, vmult, vc1))  # (stage_idx, multiplier, addend)
            else:
                vhash_multipliers.append(None)

        vn_nodes = self.alloc_scratch("vn_nodes", VLEN)
        self.add("valu", ("vbroadcast", vn_nodes, self.scratch["n_nodes"]))
        
        vforest_p = self.alloc_scratch("vforest_p", VLEN)
        self.add("valu", ("vbroadcast", vforest_p, self.scratch["forest_values_p"]))

        n_vector_batches = batch_size // VLEN  # 32
        
        # Keep ALL indices and values in scratch across rounds
        all_vidx = [self.alloc_scratch(f"all_vidx{i}", VLEN) for i in range(n_vector_batches)]
        all_vval = [self.alloc_scratch(f"all_vval{i}", VLEN) for i in range(n_vector_batches)]
        
        # Temps for pipelining
        PIPE = 4
        vnode_val_p = [self.alloc_scratch(f"vnode_val_p{i}", VLEN) for i in range(PIPE)]
        vtmp1_p = [self.alloc_scratch(f"vtmp1_p{i}", VLEN) for i in range(PIPE)]
        vtmp2_p = [self.alloc_scratch(f"vtmp2_p{i}", VLEN) for i in range(PIPE)]
        vaddr_p = [self.alloc_scratch(f"vaddr_p{i}", VLEN) for i in range(PIPE)]

        # Memory address trackers
        addr_idx = self.alloc_scratch("addr_idx")
        addr_val = self.alloc_scratch("addr_val")
        
        self.add_instr({"alu": [
            ("+", addr_idx, self.scratch["inp_indices_p"], zero_const),
            ("+", addr_val, self.scratch["inp_values_p"], zero_const),
        ]})
        
        # Load all indices and values into scratch
        for b in range(n_vector_batches):
            if b > 0:
                self.add_instr({"flow": [("add_imm", addr_idx, addr_idx, VLEN)]})
                self.add_instr({"flow": [("add_imm", addr_val, addr_val, VLEN)]})
            self.add_instr({"load": [
                ("vload", all_vidx[b], addr_idx),
                ("vload", all_vval[b], addr_val),
            ]})

        self.add("flow", ("pause",))

        round_counter = self.alloc_scratch("round_counter")
        cond = self.alloc_scratch("cond")
        rounds_const = self.scratch_const(rounds)

        # Fully unroll rounds to eliminate loop overhead
        for r in range(rounds):
            A, B, C, D = 0, 1, 2, 3
            
            # Prologue: scatter-gather for first pair, overlap address comp for second pair
            self.add_instr({"valu": [
                ("+", vaddr_p[A], all_vidx[0], vforest_p),
                ("+", vaddr_p[B], all_vidx[1], vforest_p),
                ("+", vaddr_p[C], all_vidx[2], vforest_p),
                ("+", vaddr_p[D], all_vidx[3], vforest_p),
            ]})
            # Load batches 0,1 then start loading 2,3
            for i in range(VLEN):
                self.add_instr({"load": [
                    ("load_offset", vnode_val_p[A], vaddr_p[A], i),
                    ("load_offset", vnode_val_p[B], vaddr_p[B], i),
                ]})
            self.add_instr({"valu": [
                ("^", all_vval[0], all_vval[0], vnode_val_p[A]),
                ("^", all_vval[1], all_vval[1], vnode_val_p[B]),
            ]})
            
            # Steady state: hash[N] overlapped with scatter-gather[N+1]
            for b in range(0, n_vector_batches - 2, 2):
                next_b = b + 2
                vidx_a, vval_a = all_vidx[b], all_vval[b]
                vidx_b, vval_b = all_vidx[b + 1], all_vval[b + 1]
                vidx_c, vval_c = all_vidx[next_b], all_vval[next_b]
                vidx_d, vval_d = all_vidx[next_b + 1], all_vval[next_b + 1]
                
                # Hash stage 0 using multiply_add: val = val * (1 + 2^12) + const
                # Combined with address computation for next batch
                _, vmult0, vconst0 = vhash_multipliers[0]
                self.add_instr({"valu": [
                    ("multiply_add", vval_a, vval_a, vmult0, vconst0),
                    ("multiply_add", vval_b, vval_b, vmult0, vconst0),
                    ("+", vaddr_p[C], vidx_c, vforest_p),
                    ("+", vaddr_p[D], vidx_d, vforest_p),
                ]})
                
                # Hash stages 1-5 overlapped with scatter-gather
                # Stages 2 and 4 use multiply_add optimization
                load_idx = 0  # Start from load 0
                for hi in range(1, 6):
                    vc1, vc3 = vhash_consts[hi]
                    op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                    
                    # Check if this stage can use multiply_add
                    if vhash_multipliers[hi] is not None:
                        _, vmult, vconst = vhash_multipliers[hi]
                        instr = {"valu": [
                            ("multiply_add", vval_a, vval_a, vmult, vconst),
                            ("multiply_add", vval_b, vval_b, vmult, vconst),
                        ]}
                        # Add 2 loads since we're only using 1 cycle for this stage
                        if load_idx < VLEN:
                            instr["load"] = [
                                ("load_offset", vnode_val_p[C], vaddr_p[C], load_idx),
                                ("load_offset", vnode_val_p[D], vaddr_p[D], load_idx),
                            ]
                            load_idx += 1
                        self.add_instr(instr)
                    else:
                        # Regular 2-cycle hash stage
                        instr1 = {"valu": [
                            (op1, vtmp1_p[A], vval_a, vc1),
                            (op3, vtmp2_p[A], vval_a, vc3),
                            (op1, vtmp1_p[B], vval_b, vc1),
                            (op3, vtmp2_p[B], vval_b, vc3),
                        ]}
                        if load_idx < VLEN:
                            instr1["load"] = [
                                ("load_offset", vnode_val_p[C], vaddr_p[C], load_idx),
                                ("load_offset", vnode_val_p[D], vaddr_p[D], load_idx),
                            ]
                            load_idx += 1
                        self.add_instr(instr1)
                        
                        instr2 = {"valu": [(op2, vval_a, vtmp1_p[A], vtmp2_p[A]), (op2, vval_b, vtmp1_p[B], vtmp2_p[B])]}
                        if load_idx < VLEN:
                            instr2["load"] = [
                                ("load_offset", vnode_val_p[C], vaddr_p[C], load_idx),
                                ("load_offset", vnode_val_p[D], vaddr_p[D], load_idx),
                            ]
                            load_idx += 1
                        self.add_instr(instr2)
                
                while load_idx < VLEN:
                    self.add_instr({"load": [
                        ("load_offset", vnode_val_p[C], vaddr_p[C], load_idx),
                        ("load_offset", vnode_val_p[D], vaddr_p[D], load_idx),
                    ]})
                    load_idx += 1
                
                # Index computation + XOR for next
                # Using multiply_add: idx = idx * 2 + (1 + (val & 1))
                self.add_instr({"valu": [
                    ("&", vtmp1_p[A], vval_a, vone),
                    ("&", vtmp1_p[B], vval_b, vone),
                    ("^", vval_c, vval_c, vnode_val_p[C]),
                    ("^", vval_d, vval_d, vnode_val_p[D]),
                ]})
                # Pack ADD with address computation for next-next batch if available
                next_next_b = next_b + 2
                if next_next_b + 1 < n_vector_batches:
                    self.add_instr({"valu": [
                        ("+", vtmp1_p[A], vtmp1_p[A], vone),
                        ("+", vtmp1_p[B], vtmp1_p[B], vone),
                        ("+", vaddr_p[A], all_vidx[next_next_b], vforest_p),
                        ("+", vaddr_p[B], all_vidx[next_next_b + 1], vforest_p),
                    ]})
                else:
                    self.add_instr({"valu": [
                        ("+", vtmp1_p[A], vtmp1_p[A], vone),
                        ("+", vtmp1_p[B], vtmp1_p[B], vone),
                    ]})
                self.add_instr({"valu": [
                    ("multiply_add", vidx_a, vidx_a, vtwo, vtmp1_p[A]),
                    ("multiply_add", vidx_b, vidx_b, vtwo, vtmp1_p[B]),
                ]})
                self.add_instr({"valu": [
                    ("<", vtmp1_p[A], vidx_a, vn_nodes),
                    ("<", vtmp1_p[B], vidx_b, vn_nodes),
                ]})
                self.add_instr({"valu": [
                    ("*", vidx_a, vidx_a, vtmp1_p[A]),
                    ("*", vidx_b, vidx_b, vtmp1_p[B]),
                ]})
                
                A, B, C, D = C, D, A, B
            
            # Epilogue: finish last batch pair
            last_b = n_vector_batches - 2
            vidx_a, vval_a = all_vidx[last_b], all_vval[last_b]
            vidx_b, vval_b = all_vidx[last_b + 1], all_vval[last_b + 1]
            
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                vc1, vc3 = vhash_consts[hi]
                # Use multiply_add for optimizable stages
                if vhash_multipliers[hi] is not None:
                    _, vmult, vconst = vhash_multipliers[hi]
                    self.add_instr({"valu": [
                        ("multiply_add", vval_a, vval_a, vmult, vconst),
                        ("multiply_add", vval_b, vval_b, vmult, vconst),
                    ]})
                else:
                    self.add_instr({"valu": [
                        (op1, vtmp1_p[A], vval_a, vc1),
                        (op3, vtmp2_p[A], vval_a, vc3),
                        (op1, vtmp1_p[B], vval_b, vc1),
                        (op3, vtmp2_p[B], vval_b, vc3),
                    ]})
                    self.add_instr({"valu": [(op2, vval_a, vtmp1_p[A], vtmp2_p[A]), (op2, vval_b, vtmp1_p[B], vtmp2_p[B])]})
            
            # Using multiply_add: idx = idx * 2 + (1 + (val & 1))
            self.add_instr({"valu": [
                ("&", vtmp1_p[A], vval_a, vone),
                ("&", vtmp1_p[B], vval_b, vone),
            ]})
            self.add_instr({"valu": [("+", vtmp1_p[A], vtmp1_p[A], vone), ("+", vtmp1_p[B], vtmp1_p[B], vone)]})
            self.add_instr({"valu": [("multiply_add", vidx_a, vidx_a, vtwo, vtmp1_p[A]), ("multiply_add", vidx_b, vidx_b, vtwo, vtmp1_p[B])]})
            self.add_instr({"valu": [("<", vtmp1_p[A], vidx_a, vn_nodes), ("<", vtmp1_p[B], vidx_b, vn_nodes)]})
            self.add_instr({"valu": [("*", vidx_a, vidx_a, vtmp1_p[A]), ("*", vidx_b, vidx_b, vtmp1_p[B])]})
        
        # Store final values
        self.add_instr({"alu": [("+", addr_val, self.scratch["inp_values_p"], zero_const)]})
        for b in range(n_vector_batches):
            if b > 0:
                self.add_instr({"flow": [("add_imm", addr_val, addr_val, VLEN)]})
            self.add_instr({"store": [("vstore", addr_val, all_vval[b])]})

        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
