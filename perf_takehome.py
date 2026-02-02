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

from typing import Literal
from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    cdiv,
    Instruction,
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
        self.instrs : list[Instruction] = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

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
            self.const_map[val] = addr
        return self.const_map[val]

    def store(self):
        instr = {
                "alu": [
                    ("+", self.scratch["store_addr"], self.scratch["inp_values_p"], self.zero_const),
                ]
            }
        self.instrs.append(instr)
            
        for i in range(self.n_groups):
                # Store values to mem[store_addr]
            instr = {
                "store": [
                    ("vstore", self.scratch["store_addr"], self.group_addrs[i][1]),
                ],
                "alu": [
                    ("+", self.scratch["store_addr"], self.scratch["store_addr"], self.VLEN_const),  # Increment for next group
                ]
            }
            self.instrs.append(instr)

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
            "extra_room",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        zero_const = self.alloc_scratch("zero_const", 1)
        instr =  {
            "load": [
                ("vload", self.scratch["rounds"], -1),
                ("const", zero_const, 0),
            ]
        }
        self.instrs.append(instr)
        one_const = self.alloc_scratch("one_const", 1)
        two_const = self.alloc_scratch("two_const", 1)
        instr = {
            "load": [
                ("const", one_const, 1),
                ("const", two_const, 2),
            ]   
        }
        self.instrs.append(instr)
        #Broadcast Constants
        one_vector = self.alloc_scratch("one_vector", VLEN)
        two_vector = self.alloc_scratch("two_vector", VLEN)
        n_nodes_vector = self.alloc_scratch("n_nodes_vector", VLEN)
        VLEN_const = self.alloc_scratch("VLEN_const", 1)
        tmp_var = self.alloc_scratch("tmp_var", 1)
        idx_pointer = self.alloc_scratch("idx_pointer", 1)
        value_pointer = self.alloc_scratch("value_pointer", 1)
        instr = {
            "valu": [
                ("vbroadcast", one_vector, one_const),
                ("vbroadcast", two_vector, two_const),
                ("vbroadcast", n_nodes_vector, self.scratch["n_nodes"]),
            ],
            "load":
            [
                ("const", VLEN_const, VLEN),
                ("const", tmp_var, 0),
            ],
            "alu": [
                ("+", idx_pointer, self.scratch["inp_indices_p"], zero_const),
                ("+", value_pointer, self.scratch["inp_values_p"], zero_const),
            ]
        }
        self.instrs.append(instr)


        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.instrs.append({"flow": [("pause",)]})
        # Any debug engine instruction is ignored by the submission simulator
        self.instrs.append({"debug": [("comment", "Starting loop")]})

        # Load Tree Value for first three rounds(VLEN values)
        tree_value_addrs = self.alloc_scratch("tree_values", VLEN)
        tree_level_pointer = self.alloc_scratch("tree_level_pointer", 1)
        instr = {
            "load": [
                ("vload", tree_value_addrs, self.scratch["forest_values_p"]),
                ("const", tree_level_pointer, 0),
            ]
        }
        self.instrs.append(instr)
        addr_tmp = self.alloc_scratch("addr_tmp", VLEN)
        forest_values_p_vec = self.alloc_scratch("forest_values_p_vec", VLEN)
        instr = {
            "valu": [
                ("vbroadcast", forest_values_p_vec, self.scratch["forest_values_p"]),
                ("-", addr_tmp, one_vector, one_vector)
            ]
        }
        self.instrs.append(instr)
        for i in range(SLOT_LIMITS["valu"]):
            self.alloc_scratch(f"valu_tmp_{i}", VLEN)

        for i, (op1, val1, op2, op3, val2) in enumerate(HASH_STAGES):
           self.alloc_scratch(f"hash_{i}_val1", VLEN)
           self.alloc_scratch(f"hash_{i}_val2", VLEN)
           instr = {
            "load":[
                ("const", self.scratch[f"hash_{i}_val1"], val1),
                ("const", self.scratch[f"hash_{i}_val2"], val2),
            ]
           }
           self.instrs.append(instr)
           instr = {
            "valu": [
                ("vbroadcast", self.scratch[f"hash_{i}_val1"], self.scratch[f"hash_{i}_val1"]),
                ("vbroadcast", self.scratch[f"hash_{i}_val2"], self.scratch[f"hash_{i}_val2"]),
            ]   
           }
           self.instrs.append(instr)
        #Split Batch into groups of VLEN
        self.n_groups = cdiv(batch_size, VLEN)
        self.group_addrs = {}
        for i in range(self.n_groups):
            self.group_addrs[i] = (
                self.alloc_scratch(f"group_{i}_indices", VLEN),
                self.alloc_scratch(f"group_{i}_values", VLEN),
                self.alloc_scratch(f"group_{i}_tree_values", VLEN),
            )
        #Load Input
        for i in range(self.n_groups):
            self.instrs.append(
                {
                "alu": [
                    ("+", idx_pointer, idx_pointer, VLEN_const),
                    ("+", value_pointer, value_pointer, VLEN_const),
                ],
                "valu":[
                    ("vbroadcast", self.group_addrs[i][2], tree_value_addrs),
                ],
                "load": [
                    ("vload", self.group_addrs[i][0], idx_pointer),
                    ("vload", self.group_addrs[i][1], value_pointer),
                ]
            }) 
        for round in range(rounds):
            for i in range(0, self.n_groups, SLOT_LIMITS["valu"]):
                instr = {
                    "valu": [
                        ("^", self.group_addrs[i+j][1], self.group_addrs[i+j][1], self.group_addrs[i+j][2])
                     for j in range(SLOT_LIMITS["valu"]) if i + j < self.n_groups
                ]
                }
                self.instrs.append(instr)
            # Starting Hashing
            for hash_stage_idx, (op1, _, op2, op3, _) in enumerate(HASH_STAGES):
                for i in range(0, self.n_groups, SLOT_LIMITS["valu"]):
                    #Hash part 1
                    instr = {
                        "valu": [
                            (op3, self.scratch[f"valu_tmp_{j}"], self.group_addrs[i+j][1], self.scratch[f"hash_{hash_stage_idx}_val2"])
                         for j in range(SLOT_LIMITS["valu"]) if i + j < self.n_groups
                         ]
                    }
                    self.instrs.append(instr)
                    #Hash part 2
                    instr = {
                        "valu":[
                            (op1, self.group_addrs[i+j][1], self.group_addrs[i+j][1], self.scratch[f"hash_{hash_stage_idx}_val1"])
                            for j in range(SLOT_LIMITS["valu"]) if i + j < self.n_groups
                        ]
                    }
                    self.instrs.append(instr)
                    #Hash part 3
                    instr = {
                        "valu": [
                            (op2, self.group_addrs[i+j][1], self.group_addrs[i+j][1], self.scratch[f"valu_tmp_{j}"])
                            for j in range(SLOT_LIMITS["valu"]) if i + j < self.n_groups
                        ]
                    }
                    self.instrs.append(instr)

            #Find next indices
            for i in range(0, self.n_groups, SLOT_LIMITS["valu"]):
                #2*idx + 1
                instr = {
                    "valu": [
                        ("multiply_add", self.group_addrs[i+j][0], self.group_addrs[i+j][0], two_vector, one_vector)
                     for j in range(SLOT_LIMITS["valu"]) if i + j < self.n_groups
                    ]
                }
                self.instrs.append(instr)
                # tmp = val&1
                instr = {
                    "valu": [
                        ("&", self.scratch[f"valu_tmp_{j}"], self.group_addrs[i+j][1], one_vector)
                     for j in range(SLOT_LIMITS["valu"]) if i + j < self.n_groups
                    ]
                }
                self.instrs.append(instr)
                # idx = idx + tmp
                instr = {
                    "valu":
                    [
                        ("+", self.group_addrs[i+j][0], self.group_addrs[i+j][0], self.scratch[f"valu_tmp_{j}"])
                        for j in range(SLOT_LIMITS["valu"]) if i + j < self.n_groups
                    ]
                } 
                self.instrs.append(instr)
                # wrap to zero in greater than number of tree values
                # tmp = idx < n_nodes
                # idx  = idx * tmp
                instr = {
                    "valu": [
                        ("<", self.scratch[f"valu_tmp_{j}"], self.group_addrs[i+j][0], n_nodes_vector)
                     for j in range(SLOT_LIMITS["valu"]) if i + j < self.n_groups
                    ]
                }
                self.instrs.append(instr)
                instr = {
                    "valu": [
                        ("*", self.group_addrs[i+j][0], self.group_addrs[i+j][0], self.scratch[f"valu_tmp_{j}"])
                     for j in range(SLOT_LIMITS["valu"]) if i + j < self.n_groups
                    ]
                }
                self.instrs.append(instr)
            # Load Tree Value for all groups - OPTIMIZED
            # Use multiple address buffers to pipeline address computation with loads
            # We have 6 valu slots and 2 load slots per cycle
            
            N_ADDR_BUFS = SLOT_LIMITS["valu"]  # 6 address buffers
            LOADS_PER_CYCLE = SLOT_LIMITS["load"]  # 2
            
            # Allocate address buffers once (outside rounds loop would be better, but kept here for clarity)
            if round == 0:
                self.addr_bufs = [self.alloc_scratch(f"addr_buf_{b}", VLEN) for b in range(N_ADDR_BUFS)]
            print("Before Address Computation", len(self.instrs))
            # Process groups in batches of N_ADDR_BUFS
            for batch_start in range(0, self.n_groups, N_ADDR_BUFS):
                batch_end = min(batch_start + N_ADDR_BUFS, self.n_groups)
                batch_size = batch_end - batch_start
                
                # Compute addresses for entire batch in one cycle (up to 6 valu ops)
                valu_ops = []
                for b in range(batch_size):
                    group_idx = batch_start + b
                    valu_ops.append(("+", self.addr_bufs[b], forest_values_p_vec, self.group_addrs[group_idx][0]))
                self.instrs.append({"valu": valu_ops})
                
                # Load tree values for all groups in batch
                # Interleave loads from different groups to maximize throughput
                for offset in range(0, VLEN, LOADS_PER_CYCLE):
                    for b in range(batch_size):
                        group_idx = batch_start + b
                        load_slots = []
                        for k in range(LOADS_PER_CYCLE):
                            if offset + k < VLEN:
                                load_slots.append(
                                    ("load_offset", self.group_addrs[group_idx][2], self.addr_bufs[b], offset + k)
                                )
                        self.instrs.append({"load": load_slots})
            #Store output value
            print("After Broadcast", len(self.instrs))
        #Store values in memory for stage matching
        instr = {
            "alu":[
                ("+", value_pointer, self.scratch["inp_values_p"], zero_const),
                ("+", idx_pointer, self.scratch["inp_indices_p"], zero_const)
            ]
        }
        self.instrs.append(instr)
        for i in range(self.n_groups):
            instr = {
                "alu":[
                    ("+", value_pointer, value_pointer, VLEN_const),
                    ("+", idx_pointer, idx_pointer, VLEN_const)
                ],
                "store": [
                    ("vstore", value_pointer, self.group_addrs[i][1]),
                    ("vstore", idx_pointer, self.group_addrs[i][0])
                ]
            }
            self.instrs.append(instr)
        #Pause
        self.instrs.append({"flow":[("pause",)]})
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
    # print(mem)

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
        ), f"Incorrect result on round {i}, {machine.mem[inp_values_p : inp_values_p + len(inp.values)]} != {ref_mem[inp_values_p : inp_values_p + len(inp.values)]}"
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
            inp = Input.generate(f, 16, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)], f"{inp.indices} != {mem[mem[5] : mem[5] + len(inp.indices)]}"
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)], f"{inp.values} != {mem[mem[6] : mem[6] + len(inp.values)]}"

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=False, prints=False)

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
