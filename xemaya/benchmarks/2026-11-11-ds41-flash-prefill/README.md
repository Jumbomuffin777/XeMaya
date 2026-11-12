<!--
SPDX-FileCopyrightText: 2026 Jumbomuffin777 and the XeMaya contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# DeepSeek-V4.1-Flash (regular, 384 experts) IQ2_XXS: prefill on B70 + B65 (2026-11-11)

Speed only; the generated text is compared between the arms, not against a reference.

- **Model:** DeepSeek-V4.1-Flash-1017-GSQ-RCO IQ2_XXS, the regular 384-expert file (123 GiB of routed experts), not the
  Coder file. Its shards are on a PCIe Gen 5 NVMe SSD (32 GT/s x4), not the Gen 3 drive of
  [2026-11-10](../2026-11-10-ds41-flash-384/README.md): the two days' numbers are not comparable.
- **Host:** Arc Pro B70 + B65 (`--layer-split auto`: blocks 0-19 / 20-39), 120 GiB RAM, the engine in a memory cap of
  80 GiB. The model is larger than the VRAM and than the RAM: every prompt starts with an empty VRAM tier and reads
  its experts from the files (`--mmap-experts`).
- **Build:** the engine at this commit, Intel oneAPI icpx 2026.2. The free and contrib-llvm modes with intel/llvm, the
  Windows cross build, a smaller GPU and other makers' GPUs are unverified.
- **Settings:** `--greedy --max-context 2048 / 4096 / 10240 --layer-split auto --mmap-experts --expert-cache auto
  --max-new 64 --spec 0`; prompts of 576 tokens (`flash-mid`), 2,048 and 8,192 tokens (this repository's documents and
  sources). One process a cell, each through `gpu_job.sh` (lease, safety guard): status 0, no kernel message.

## Where the time was (576 tokens, `STRATA_PROMPT_PROFILE=1`, before the change)

28.1 s: the routed experts 20.0 s (B70 9.9, B65 10.1), the VRAM tiers' fill after the prompt 5.6 s (4,745 experts),
everything else 2.4 s (the attention 0.17 s, its projections and output 0.39 s, the mHC 0.15 s). Of the experts'
20.0 s, 0.6 s was waiting for the files and 20.0 s the prompt thread's copy of each expert (8.4 MiB, 9,874 of them)
from the source's buffer into the pinned staging: one thread's copy beside eight reading threads, 4.3 GB/s. Neither
GPU's arithmetic nor the drive was the limit; the two cards take the same time.

## What changed

1. The read-ahead threads read each expert from the files straight into its place in the pinned staging
   (`ExpertSource::read_blobs`); `XEMAYA_PROMPT_PINNED_READS=0` keeps the copy.
2. The tiers' fill after the prompt reads 16 experts at a time side by side and queues their copies
   (`PlainModel::fill_slots`); `XEMAYA_TIER_FILL_BATCH=0` fills one at a time. The same experts go into the same slots.
3. Opt-in, `XEMAYA_EXPERT_UNBUFFERED=1`: the prompt's reads past the file cache (Linux, `O_DIRECT`).

## Results (prefill tok/s; the 63 tokens after it in parentheses)

| Prompt | Before (1 and 2 off) | Default now (1 and 2) | With `XEMAYA_EXPERT_UNBUFFERED=1` |
| --- | ---: | ---: | ---: |
| 576 tokens | 20.0 (12.8) | **27.5** (13.1) | 32.8 (9.9) |
| 2,048 tokens | 36.5 (11.3) | **47.4** (11.4) | 60.3 (8.8) |
| 8,192 tokens | 35.2 (8.4) | **46.2** (8.7) | 60.8 (9.1) |

Prefill seconds: 28.8 / 21.0 / 17.6, 56.1 / 43.2 / 34.0 and 232.7 / 177.1 / 134.7. The generated tokens are the same
in the three arms at every length. A second default run at 576 tokens: 26.7 tok/s (12.8).

- With 1 and 2 the experts' 20.0 s are 15.3 s, all of it waiting for the files (5.6 GB/s through the file cache), and
  the tiers' fill 4.7 s.
- Unbuffered, the experts take 10.9 s (7.9 GB/s; the drive alone gives 6 to 11.5 GiB/s to
  `unbuffered_read_test --bench`) and the fill 5.8 s. It is not the default: the file cache is then cold for the
  experts the decode's CPU pool reads, and the tokens after a 576- or 2,048-token prompt come 23% slower (equal after
  8,192). The whole run is shorter with it at every length measured (576 tokens and 64 generated: 23.9 s against
  25.8 s; 8,192: 141.6 s against 184.4 s).
- Unbuffered reads alone, before 1 (the copy still in place), gained nothing: 20.9 s for the experts against 18.1 s.

## What limits it now

Every chunk of 1,024 tokens routes to nearly all of a layer's 384 experts, and none stays in VRAM during the prompt:
a prompt reads the 123 GiB of experts once a chunk (8,192 tokens: 804 GB from the drive). The time is the drive's and
the file cache's. Next: keep what a chunk uploads in the tier's free slots (43% of the experts fit), so that later
chunks read them where they lie; and a longer chunk where the prompt's buffers allow it. A 32K prompt was not run: it
would read about 4 TB.

`ds41_parity` passes on the B70 and on the B65 with this build; `unbuffered_read_test` (475 ranges against the same
bytes read through the cache) passes.
