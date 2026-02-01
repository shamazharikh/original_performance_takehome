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
        zero_const = self.scratch_const(0)
        instr =  {
            "load": [
                ("vload", self.scratch["rounds"], -1),
                ("const", zero_const, 0),
            ]
        }
        self.instrs.append(instr)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
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
        instr = {
            "valu": [
                ("vbroadcast", one_vector, one_const),
                ("vbroadcast", two_vector, two_const),
            ]
        }
        self.instrs.append(instr)
        VLEN_const = self.scratch_const(VLEN)
        tmp_var = self.alloc_scratch("tmp_var", 1)
        instr = {
            "load":
            [
                ("const", VLEN_const, VLEN),
                ("const", tmp_var, 0),
            ]
        }
        self.instrs.append(instr)

        idx_pointer = self.alloc_scratch("idx_pointer", 1)
        value_pointer = self.alloc_scratch("value_pointer", 1)
        instr = {
            "load": [
                ("const", idx_pointer, 0),
                ("const", value_pointer, 0),
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
        n_groups = cdiv(batch_size, VLEN)

        self.group_addrs = {}
        for i in range(n_groups):
            self.group_addrs[i] = (
                self.alloc_scratch(f"group_{i}_indices", VLEN),
                self.alloc_scratch(f"group_{i}_values", VLEN),
                self.alloc_scratch(f"group_{i}_tree_values", VLEN),
            )
        #Load Input
        for i in range(n_groups):
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
            for i in range(n_groups, step=SLOT_LIMITS["valu"]):
                instr = {
                    "valu": [
                        ("^", self.group_addrs[i][1], self.group_addrs[i][1], self.group_addrs[i][2])
                     for i in range(SLOT_LIMITS["valu"])
                ]
                }
                self.instrs.append(instr)
            # Starting Hashing
            for (op1, val1, op2, op3, val2) in HASH_STAGES:
                for i in range(n_groups, step=SLOT_LIMITS["valu"]):
                    #Hash part 1
                    instr = {
                        "valu": [
                            (op3, self.scratch[f"valu_tmp_{j}"], self.group_addrs[i+j][1], self.scratch[f"hash_{j}_val2"])
                         for j in range(SLOT_LIMITS["valu"])
                    ]
                    }
                    self.instrs.append(instr)
                    #Hash part 2
                    instr = {
                        "valu":[
                            (op1, self.group_addrs[i+j][1], self.group_addrs[i+j][1], self.scratch[f"hash_{j}_val1"])
                            for j in range(SLOT_LIMITS["valu"])
                        ]
                    }
                    self.instrs.append(instr)
                    #Hash part 3
                    instr = {
                        "valu": [
                            (op2, self.group_addrs[i+j][1], self.group_addrs[i+j][1], self.scratch[f"valu_tmp_{j}"])
                            for j in range(SLOT_LIMITS["valu"])
                        ]
                    }
                    self.instrs.append(instr)

            if round % 5 == 4:
                for i in range(n_groups, step=SLOT_LIMITS["valu"]):
                    instr = {
                        "valu": [
                            ("vbroadcast", self.group_addrs[i+j][0], zero_const)
                            for j in range(SLOT_LIMITS["valu"])
                            ]
                    }
                    self.instrs.append(instr)
                for i in range(n_groups, step=SLOT_LIMITS["valu"]):
                    instr= {
                        "valu":[
                        ("vbroadcast", self.group_addrs[i+j][2], self.scratch["forest_values_p"]) 
                        for j in range(SLOT_LIMITS["valu"])
                        ]
                    }
                    self.instrs.append(instr)
                continue

            #Find next indices
            for i in range(n_groups, step=SLOT_LIMITS["valu"]):
                #2*idx + 1
                instr = {
                    "valu": [
                        ("multiply_add", self.group_addrs[i+j][0], self.group_addrs[i+j][0], two_vector, one_vector)
                     for j in range(SLOT_LIMITS["valu"])
                    ]
                }
                self.instrs.append(instr)
                # tmp = val&1
                instr = {
                    "valu": [
                        ("&", self.scratch[f"valu_tmp_{j}"], self.group_addrs[i+j][1], one_vector)
                     for j in range(SLOT_LIMITS["valu"])
                    ]
                }
                self.instrs.append(instr)
                # idx = idx + tmp
                instr = {
                    "valu":
                    [
                        ("+", self.group_addrs[i+j][0], self.group_addrs[i+j][0], self.scratch[f"valu_tmp_{j}"])
                        for j in range(SLOT_LIMITS["valu"])
                    ]
                } 
                self.instrs.append(instr)
            #Load Tree Value
            for i in range(n_groups):
                for j in range(VLEN):
                    instr = {
                        "load": [
                            ("load", self.group_addrs[i][2], self.scratch["forest_values_p"], j)
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
