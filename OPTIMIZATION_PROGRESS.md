# Kernel Optimization Progress Summary

## Current Status

| Metric | Value |
|--------|-------|
| **Baseline** | 147,734 cycles |
| **Current** | 4,718 cycles |
| **Speedup** | **31.3x** |
| **Target** | < 1,487 cycles |
| **Gap** | ~3.2x more improvement needed |

## Architecture Constraints
- **VLEN**: 8 (vector length)
- **SLOT_LIMITS**: `alu: 12`, `valu: 6`, `load: 2`, `store: 2`, `flow: 1`
- **SCRATCH_SIZE**: 1536 words
- **N_CORES**: 1

---

## Key Optimizations Applied

### 1. SIMD Vectorization
- Replaced scalar ops with vector ops (`vload`, `vstore`, `valu`)
- Process 8 elements per operation (VLEN=8)
- `vbroadcast` for constant initialization

### 2. Data in Scratch Across Rounds
- Keep all 256 indices/values in scratch memory
- Eliminates expensive memory round-trips between rounds
- Only load at start, store at end

### 3. 4-Way Software Pipelining
- Overlap scatter-gather[N+1] with hash computation[N]
- Hash stages 1-5 run in parallel with `load_offset` operations
- Used 4 register sets for double-buffering (2 current, 2 next)

Structure:
```
Prologue: scatter-gather for first batch pair, XOR
Steady State: 
  - Hash stage 0[N] + compute vaddr[N+1]
  - Hash stages 1-5[N] + scatter-gather[N+1] (overlapped)
  - Index computation[N] + XOR[N+1]
Epilogue: Complete last batch pair
```

### 4. VLIW Instruction Packing
- Combine `valu` + `load` operations in same cycle
- Use all 6 valu slots where possible (e.g., hash ops + address computation)
- Combined XOR for next batch with index computation for current batch

### 5. Full Loop Unrolling
- Fully unrolled 16 rounds to eliminate loop overhead (~48 cycles saved)
- No `cond_jump` overhead

### 6. Arithmetic Optimizations
- Replaced `1 if val%2==0 else 2` → `1 + (val & 1)`
- Bounds check via `idx * (idx < n)` instead of `vselect`

---

## Bottlenecks Remaining

### Sequential Dependencies
1. **Scatter-gather**: 8 `load_offset` per batch pair per round (partially hidden by pipelining)
2. **Hash dependency chain**: 6 stages × 2 cycles = 12 cycles minimum for 2 batches
3. **Index computation**: 5 cycles per batch pair

### Resource Utilization
- **VALU**: Well utilized during hash (4-6 ops per cycle)
- **Load**: 2 loads per cycle, fully utilized during scatter-gather
- **Store**: 2 stores per cycle, used efficiently
- **Flow**: Minimized (only for address increments)

---

## Performance History

| Optimization Stage | Cycles | Speedup |
|-------------------|--------|---------|
| Baseline (scalar) | 147,734 | 1.0x |
| SIMD + VLIW + Hardware Loops | 15,475 | 9.5x |
| Fixed slot limits | 11,633 | 12.7x |
| Eliminated vselect | 10,865 | 13.6x |
| Software pipelining (2-batch) | 5,696 | 25.9x |
| Merged XOR with index comp | 5,456 | 27.1x |
| Overlapped store with multiply | 5,216 | 28.3x |
| Data in scratch across rounds | 4,767 | 31.0x |
| Full loop unrolling | 4,718 | 31.3x |

---

## Next Steps to Explore

1. **Round 0 broadcast**: All indices start at 0 → load `tree[0]` once and broadcast (eliminates scatter-gather for round 0)
2. **Level-based tree caching**: For levels 1-5, preload tree values and use selection instead of gather
3. **Process 3 batches at once**: Better utilize 6 valu slots (currently using 4)
4. **Cross-round pipelining**: Overlap epilogue of round N with prologue of round N+1

---

## Files Modified
- `perf_takehome.py`: Main kernel implementation in `KernelBuilder.build_kernel()`

## Test Commands
```bash
python3 tests/submission_tests.py
```

---

*Last updated: Optimization session achieving 31.3x speedup (4,718 cycles)*
