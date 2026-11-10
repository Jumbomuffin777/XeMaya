# DeepSeek-V4.1-Flash (regular, 384 experts) IQ2_XXS: two vs three Arc GPUs (2026-11-10)

Speed only: **no reference comparison was run** for these results. They are separate from the Coder results
(DeepSeek-V4.1-Flash-1017-GSQ-RCO-**Coder** IQ2_XXS, 192 experts) in README and STATUS; the two files are different
models and their numbers are not interchangeable.

- **Build:** the published `0.1.0-alpha.1` release build (`strata` from the release archive, Intel oneAPI icpx 2026.2).
  Nothing was rebuilt for these runs.
- **Model:** DeepSeek-V4.1-Flash-1017-GSQ-RCO IQ2_XXS (six shards, 129 GB), the regular file with 384 routed experts a
  layer, not the Coder file. The model shards and the Engram file are on a **PCIe Gen 3 NVMe SSD** (8 GT/s x4).
- **Host:** Arc Pro B70 (32 GB) + B65 (32 GB) + B60 (24 GB; behind the chipset, 7.0 GB/s from the host against
  28.5 GB/s for the other two), 128 GB RAM. The model is larger than the cards' VRAM, so experts the VRAM tier misses
  run in the CPU pool.
- **Settings:** `--greedy --max-context 2048 --layer-split auto --mmap-experts --expert-cache auto --spec 0`, the
  576-token `flash-mid` prompt. Two GPUs: `--layer-split auto`'s default (it leaves out the slow-link B60). Three
  GPUs: `XEMAYA_SLOW_LINK_GPUS=1`.
- **Safety:** every run through `gpu_job.sh` (whole-host lease, `safety_guard.sh`, memory cap): status 0, no kernel
  message. Logs: `<work>/logs/ds41-flash-speed/` (two GPUs, 128 tokens) and `<work>/logs/ds41-flash-3gpu/`.

## Decode length 1,024 tokens (one process each, the page cache warm)

| GPUs | Layer split | Prefill (576 tokens) | Decode (1,023 generated) | VRAM tier filled | Routed experts hit | Swaps | CPU pool's share of decode windows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **B70 + B65** | blocks 0-19 / 20-39 | 11.4 tok/s (50.4 s) | **14.56 tok/s** | 6,540 of 6,540 | 93.7% | 8,056 | 58% |
| **B70 + B65 + B60** | 0-16 / 17-33 / 34-39 | 11.2 tok/s (51.2 s) | **11.58 tok/s** | 8,486 of 9,104 | 94.2% | 9,797 | 56% |

## Short runs, 127 generated tokens (cold: the model's files dropped from the page cache first)

| GPUs | Run | Load | Prefill | Decode | Hit / swaps | Peak VRAM (B70 / B65 / B60) | Engine RSS peak | Page cache peak |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| B70 + B65 | cold | 4.9 s | 11.4 tok/s | 10.55 tok/s | 91.5% / 5,832 | 29.4 / 29.5 / 0.2 GiB | 16.1 GiB | 85.9 GiB |
| B70 + B65 | warm | 6.3 s | 11.4 tok/s | 10.60 tok/s | 91.5% / 5,832 | same | 15.6 GiB | 85.9 GiB |
| B70 + B65 + B60 | cold | 5.1 s | 11.2 tok/s | 9.76 tok/s | 92.1% / 6,116 | 29.5 / 29.5 / 21.7 GiB | 16.5 GiB | 87.0 GiB |
| B70 + B65 + B60 | warm | 6.6 s | 11.2 tok/s | 9.72 tok/s | 92.1% / 6,116 | same | 17.0 GiB | 85.4 GiB |

The load time excludes the experts: with `--mmap-experts` they are read on first use, inside the prompt's time. Cold
and warm agree because every process starts with an empty VRAM tier and the 129 GB file does not stay in the page cache.

## Why two GPUs are faster than three here

- **The B60 adds little to the expert cache.** Its six blocks hold all 384 experts each (no swaps on it), and the
  B70 / B65 tiers rise from 171 / 156 to 209 / 191 slots a layer, but the hit rate moves only from 93.7% to 94.2%:
  the most-routed experts already fit on two cards.
- **The B60's blocks cost more than they save.** Decode takes ~68.7 ms a token on two GPUs and ~86.3 ms on three:
  ~17.6 ms more for the six blocks on the B60, about 2.7 times what the same share of blocks costs on the two-GPU
  path. This is the B60's own execution and the extra hand-off to and from it over its chipset link; the logs do not
  separate the two (that needs a per-device profile).
- **The CPU pool stays the same.** It takes 56-58% of the decode windows either way.
- **Prefill does not depend on the GPU count** (~11.3 tok/s): the prompt's experts are read from the files on the
  Gen 3 SSD as they are first used.

## XMX

The engine reports XMX matrix acceleration: "prompt matrix products on oneMath (oneMKL), XeStrata's own kernels (XMX)
for the rest; the experts' in int8 (XMX)". DeepSeek's XMX attention (`ds41_sink_attn_xmx`) is the default and nothing
disabled it (`XEMAYA_DS41_ATTN` and `STRATA_NO_XMX` unset), but the engine does not log the attention path it took, so
**the attention kernel path was not independently verified** in these runs.
