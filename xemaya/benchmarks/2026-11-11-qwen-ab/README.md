<!--
SPDX-FileCopyrightText: 2026 Jumbomuffin777 and the XeMaya contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Qwen3.8-Flash-Next IQ3_S decode against Strata v0.2.12, B70 + B65 (2026-11-11)

`tools/xemaya/window4_qwen.sh` (`qwen_bench.py`: the production profile's arguments, `--spec 4`, 256 tokens a
request, greedy), the same model files, prompts and settings as window 4; one load an arm. The engine is the
development build of 2026-11-10. No kernel message in any run.

| Arm | 26.8K code tok/s (drafts kept) | 26.8K prose tok/s | 59-token code | 68-token prose | 26.8K prefill tok/s |
|---|---|---|---|---|---|
| Strata v0.2.12 | 78.7 (167 of 196) | 59.2 (131 of 233) | 84.2 | 58.7 | 2,276 / 2,293 |
| XeMaya (`--mmap-experts`: the min-p window rule) | 77.3 (165 of 194) | 57.8 (128 of 234) | 80.7 | 61.2 | 1,877 / 1,939 |
| XeMaya `--spec-window auto` (the measured-cost rule) | 74.9 (163 of 203) | 59.3 (129 of 231) | 80.7 | 62.7 | 1,873 / 1,938 |

With `STRATA_DECODE_TIMING=1` (26.8K code request, per verify window): XeMaya 36.63 ms = verify 32.11 + draft 4.47
at 2.81 tokens a window; Strata v0.2.12 (window 4's record) 36.09 ms = verify 31.98 + commit 0.89 + draft 3.22 at
2.59 tokens. After a short prompt XeMaya's draft is 2.9-3.3 ms a window, Strata's 1.97: the draft costs XeMaya
about 1 ms a window more at either context.

What this settles:

- The window rule is not the gap. The measured-cost rule gains 1.5 tok/s on prose and loses 2.4 on code at 26.8K;
  the default (min-p with `--mmap-experts`) stays. With min-p XeMaya drafts and keeps what Strata does on code
  (165 of 194 against 167 of 196).
- The draft layer's existing switches do not close it: `STRATA_MTP_PRIO=1` 77.4 code / 58.1 prose,
  `STRATA_MTP_SHARED_BRANCH=0` 76.8 / 57.5 (draft 4.35 ms a window); the plain arm run again 76.8 / 57.6, so
  the run-to-run spread is about 0.5 tok/s.
- The remaining gap on code (1.4 tok/s, 1.8%) is the draft step: about 0.5 ms a draft step more than Strata's,
  the same at 59 tokens and at 26.8K, so it is a fixed cost of the step (its ~20 launches), not its attention
  over the context.

## The draft round's parts (`STRATA_MTP_TIMING=1`, the host's clock)

| Context | Round graph (the window's K/V catch-up and draft 0) | Each later step graph | Steps a round |
|---|---|---|---|
| 59 and 68 tokens | 1.49-1.52 ms, 68-79 launches | 0.98 ms, 67 launches | 1.4-1.5 |
| after 26.8K (the cumulative averages rise to) | 1.87-1.93 ms | 1.30-1.36 ms | 1.5 |

Staging before the launch is 0.025 ms and the last sync 0.04 ms: the time is in the two graphs. A step is 67
launches in about 1 ms at a short context, some 15 us a launch, so it is the number of launches that costs, and
0.3-0.6 ms more once the layer's attention reads 26.8K cells.

Ruled out: the SYCL runtime. The same engine on the runtime Strata v0.2.12 ships (oneAPI 2026.1.1; XeMaya's is
2026.2) gives the same round and step times (1.87 and 1.31 ms). oneMKL's backend does not load against that
runtime, so that run's other numbers do not compare.

## The step by part (`STRATA_MTP_PROFILE=1`: the steps run outside their graphs, a wait after each part)

1,239 steps over the whole benchmark (short and 26.8K contexts mixed), 1.41 ms a step with the waits:

| Part | ms a step | Share |
|---|---|---|
| the layer's attention over the context (`window_ids`, `qsa_decode_attn_batch`) | 0.458 | 32.5% |
| the draft head (q8, `native_mmvq` over 40,525 rows) | 0.228 | 16.2% |
| the routed experts (group, q8_0, `moe_grouped_s2`) | 0.180 | 12.8% |
| q projection, norm, rope, rotation | 0.079 | 5.6% |
| each of the 14 other parts | 0.009-0.055 | 0.6-3.9% |

So the launch count is not the cost (this corrects the reading above): three parts are 61% of the step, and the
attention is the part that grows with the context. It is the dense attention of `qsa_decode_attn.cpp`, a scalar
port of Strata's CUDA kernel; the matrix engines are not used.

`--mtp-window` (the cells the draft layer attends to; 32768 by default, as Strata's), 26.8K code / prose tok/s and
the code request's drafts kept:

| `--mtp-window` | Code | Prose | Drafts kept (code) | Draft round, ms |
|---|---|---|---|---|
| 32768 | 77.3 | 57.8 | 165 of 194 | 3.85 |
| 8192 | 77.9 | 58.8 | 162 of 196 | 3.02 |
| 4096 | 73.3 | 58.6 | 155 of 207 | 2.89 |
| 2048 | 73.8 | 59.5 | 155 of 207 | 2.80 |

A shorter window saves the time and loses drafts: 8192 is +0.6 / +1.0 tok/s (the spread between runs is about 0.5),
4096 and less lose 4 tok/s on code. The default stays.

Next: the draft layer's dense attention on the matrix engines, as `ds41_attn_xmx.cpp` and `minimax_attn_xmx.cpp`
do for their models (an online softmax over FP16 tiles), for the int8 K/V the draft layer keeps, with a parity test
against `qsa_dense_attn_*` and the scalar kernel as the fallback. The drafts are checked by the model, so the kernel
need not be bit-equal, only close enough to keep the drafts. At 26.8K the attention is about 0.8 ms a step and a
window has about 2.5 steps: a kernel four times faster shortens the window by 1.5 ms, which is what 80 tok/s on
code needs. After it, the draft head (0.23 ms a step).
