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

Not done: a per-kernel timing of the draft step (`MtpDrafter::record_forward`), which is the next measurement; 80
tok/s on code needs the window 1.5 ms shorter, more than the draft gap alone (1.0-1.25 ms).
