# XeMaya: notes for coding agents

Read this, then `PLAN.md` and `STATUS.md` (same folder). The repository's root `AGENTS.md` is XeStrata's and still
governs code conventions (hardware independence, no vendor-string decisions, licences, SPDX headers), except where this
file overrides it.

## Production safety (non-negotiable)

Zability serves production Qwen3.8 on the B70 + B65 through Strata 0.2.12 (`zability-strata-qwen38.service`). Hermes
coding agents depend on it.

- Never stop, restart or reconfigure production services, profiles, binaries or model files
  (`/DATA/AppData/strata-void-l3/...`). Read them only.
- Never run GPU work outside the lease (`/DATA/AppData/zability-gpu-lease/gpulease.py`) and preflight
  (`/DATA/AppData/strata-void-l3/preflight.sh`). Never edit those tools. While production holds the whole-host lease,
  only the owners in `gpu_partitions.py` (`comfyui`, `marrow-qa` on the B60) may claim a card. XeMaya has no owner yet:
  GPU tests wait for the owner's approval.
- Earlier DeepSeek tests on this host caused two host OOMs that killed production services. Run every build and test
  under `systemd-run --user --scope -p MemoryMax=... -p MemorySwapMax=0 nice -n 19 ionice -c3`, with `ninja -j4`.
- No changes to networking, Docker, Hermes, FrontierUI, MARROW or firewall.
- Nothing is published or pushed to anyone else's repository. Pushing to the private `Jumbomuffin777/XeMaya` is allowed.

## Where things are

| What | Where |
|---|---|
| Repo (branch `xemaya/main`, pushes to `origin main`) | `/DATA/Projects/xemaya/XeMaya` |
| Remotes | `origin` = Jumbomuffin777/XeMaya (private), `xestrata` = MistVVK/XeStrata, `strata` = Niko1221/Strata |
| Maya reference | `/DATA/Projects/xemaya/ref/project-maya` |
| llama.cpp reference graphs (read only) | `/DATA/AppData/maya-upstream/llama.cpp-master-20261029/src/models/{deepseek41,deepseek4,glm5-next}.cpp` |
| Builds | `/DATA/Projects/xemaya/build-icpx` (tests, no ggml), `/DATA/Projects/xemaya/build-engine` (the `strata` engine) |
| Python venv for serve tests | `/DATA/Projects/xemaya/venv` |
| Logs | `/DATA/Projects/xemaya/logs`, `/DATA/Projects/xemaya/*.log` |
| Models | `/mnt/ai/models/ISTA-DASLab/` (DeepSeek, Qwen), `/mnt/ai/models/peasantsmith/GLM-5.3-Flash-Maya-GGUF/` (GLM) |
| Prior Intel work and its lessons | `/DATA/AppData/maya-upstream/README.md`, `native-port/AUDIT.md` |

Access: from the Windows PC, `ssh zability` works directly. `gh` is authenticated on Zability (Jumbomuffin777), not on
Windows. Send scripts with `scp` and run them. PowerShell mangles `$(...)` inside double-quoted ssh arguments.

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
Porting an upstream serve commit: `python3 /DATA/Projects/xemaya/conf.py show <commit>` prints its conflict hunks,
`resolve <spec.json>` applies per-hunk choices (`ours`, `theirs`, `both`, or `{"text": ...}`), and `finish <commit>
[tests]` runs the commit's tests plus the listed ones and commits only if they pass.

Model headers without downloading: `tools/xemaya/gguf_inspect.py hf:OWNER/REPO/PATH.gguf`.
Model shape as the engine reads it: `build-icpx/model_spec_test FILE.gguf`.

## Conventions

- XeMaya commit messages are in English. Commit one logical change at a time, and say in the message what was checked.
- New XeMaya files carry `SPDX-FileCopyrightText: 2026 the XeMaya contributors` and
  `SPDX-License-Identifier: LGPL-3.0-or-later`. Code transcribed from llama.cpp names the commit and its MIT licence.
- Mark anything not run on real hardware `unverified` in STATUS.md. Never report a measurement that was not taken.
- Keep STATUS.md current before ending a session.
