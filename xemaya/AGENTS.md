<!--
SPDX-FileCopyrightText: 2026 Jumbomuffin777 and the XeMaya contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# XeMaya: notes for coding agents

Read this, then `PLAN.md` and `STATUS.md` (same folder). The repository's root `AGENTS.md` is XeStrata's and still
governs code conventions (hardware independence, no vendor-string decisions, licences, SPDX headers), except where this
file overrides it.

## Production safety (non-negotiable)

The development host also serves a production model on the B70 + B65 through Strata 0.2.12, and other work depends
on it. The host's names, folders, units and tools are in `LOCAL.md` (git ignores it; the root `AGENTS.md` says what
goes there); the placeholders below (`<work>`, `<production>`, `<lease>`, `<models>`) are its entries.

- Never stop, restart or reconfigure production services, profiles, binaries or model files
  (`<production>/...`). Read them only.
- Never run GPU work outside the lease (`<lease>/gpulease.py`) and preflight
  (`<production>/preflight.sh`). Never edit those tools. While production holds the whole-host lease,
  only the owners its partition table names may claim a card. XeMaya has no owner there: GPU tests wait for the
  owner's approval.
- Earlier DeepSeek tests on this host caused two host OOMs that killed production services. Run every build and test
  under `systemd-run --user --scope -p MemoryMax=... -p MemorySwapMax=0 nice -n 19 ionice -c3`, with `ninja -j4`.
- No changes to the host's networking, containers, firewall or other services.
- **Host stability (2026-11-09 incident, STATUS.md):** the B60 sits behind the chipset with both NICs; sustained
  streaming to it made the NIC time out and the host lose its network. Do not use the B60 for XeMaya unless the owner
  says so (`--layer-split auto` now leaves it out). Run every GPU job under `tools/xemaya/safety_guard.sh`. Never
  SIGKILL an engine in the middle of GPU work (QUIT, Ctrl-C, SIGTERM and wait). Check `journalctl -k` for
  `NETDEV WATCHDOG` and `xe ... Engine reset` after any GPU run, and stop at the first one. Think through PCIe, disk,
  memory and network load before a long run, and keep long-context runs short and few.
- Nothing is pushed to anyone else's repository. Pushing to `Jumbomuffin777/XeMaya` is allowed.

## Where things are

| What | Where |
|---|---|
| Repo (branch `xemaya/main`, pushes to `origin main`) | `<work>/XeMaya` |
| Remotes | `origin` = Jumbomuffin777/XeMaya, `xestrata` = MistVVK/XeStrata, `strata` = Niko1221/Strata |
| Maya reference | `<work>/ref/project-maya` |
| llama.cpp reference graphs (read only) | `<maya-upstream>/llama.cpp-master-20261029/src/models/{deepseek41,deepseek4,glm5-next}.cpp` |
| Builds | `<work>/build-icpx` (tests, no ggml), `<work>/build-engine` (the `strata` engine) |
| Python venv for serve tests | `<work>/venv` |
| Logs | `<work>/logs`, `<work>/*.log` |
| Models | `<models>/ISTA-DASLab/` (DeepSeek, Qwen), `<models>/peasantsmith/GLM-5.3-Flash-Maya-GGUF/` (GLM) |
| Prior Intel work and its lessons | `<maya-upstream>/README.md`, `native-port/AUDIT.md` |

Access to the development host and where `gh` is authenticated: `LOCAL.md`.

## Build and test

```sh
source /opt/intel/oneapi/setvars.sh --force
# tests and kernels, no model
cmake -S XeMaya -B build-icpx -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx \
  -DSTRATA_LICENSE=contrib -DSTRATA_CUDA_ARCHS= -DSTRATA_HIP_ARCHS= -DSTRATA_ENABLE_XE=ON \
  -DSTRATA_NATIVE_EXPERTS=OFF -DSTRATA_BUILD_TESTS=ON
# the engine: add -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_GGML_DIR=$PWD/XeMaya/third_party/main/llama.cpp (pinned 3cf03257)
systemd-run --user --scope -q -p MemoryMax=28G -p MemorySwapMax=0 nice -n 19 ionice -c3 ninja -C build-engine -j4 strata
```

Kernel tests without a GPU: `ONEAPI_DEVICE_SELECTOR=opencl:cpu XEMAYA_CPU_DEVICE=1 ./glm5_parity`. The selector hides
every GPU from SYCL, so nothing touches the leased cards. Tests that register host memory for device copies
(`elementwise_parity`) need Level Zero and run only on a GPU.

The same for DeepSeek: `ONEAPI_DEVICE_SELECTOR=opencl:cpu XEMAYA_CPU_DEVICE=1 build-icpx/ds41_parity`. A real
model's tensors: `build-icpx/ds41_manifest_test <any shard>`.

Serve tests: `venv/bin/python -m pytest -q -p no:cacheprovider serve` (930 passed, 23 skipped after the ports).
Porting an upstream serve commit: `python3 <work>/conf.py show <commit>` prints its conflict hunks,
`resolve <spec.json>` applies per-hunk choices (`ours`, `theirs`, `both`, or `{"text": ...}`), and `finish <commit>
[tests]` runs the commit's tests plus the listed ones and commits only if they pass.

GPU windows (owner-approved, production stopped by the owner): `tools/xemaya/gpu_window.sh` under
`gpulease.py run --holder xemaya-dev` (STATUS.md has the command); it compares against a reference made beforehand on
the CPU with `tools/xemaya/ref_cpu.sh` (llama.cpp v0.8.0 built CPU-only in `<work>/ref/llama.cpp-v0.8.0`)
through `tools/xemaya/compare_ref.py`. `iq_parity` reads its fixtures from `<work>/iq_fixture`
(`tools/iq_fixture.py`); on the CPU device it takes ~35 minutes.

GLM serving pieces: the tokenizer folder `<work>/serve-glm/tok/tokenizer` (from
`tools/strata_tokenizer.py`), the window prompts in `<work>/prompts` (`tools/xemaya/make_prompts.py`).
`XEMAYA_MLA_ATTN=tiled` selects the tiled DSA attention; `XEMAYA_PROMPT_CHECKPOINT=0` turns the prompt-end
checkpoint off. `tools/xemaya/engine_serve.py` drives a loaded engine with many requests (one load per window phase).

Model headers without downloading: `tools/xemaya/gguf_inspect.py hf:OWNER/REPO/PATH.gguf`.
Model shape as the engine reads it: `build-icpx/model_spec_test FILE.gguf`.

## Conventions

- XeMaya commit messages are in English. Commit one logical change at a time, and say in the message what was checked.
- New XeMaya files carry the SPDX header of the root `AGENTS.md` with the holder `2026 Jumbomuffin777 and the XeMaya
  contributors` (earlier files say `2026 the XeMaya contributors`) under the LGPL, version 3 or later. Code
  transcribed from llama.cpp names the commit and its MIT licence.
- Mark anything not run on real hardware `unverified` in STATUS.md. Never report a measurement that was not taken.
- Keep STATUS.md current before ending a session.
