<!--
SPDX-FileCopyrightText: 2026 Jumbomuffin777 and the XeMaya contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# Installing XeMaya (Linux, Intel Arc)

This is the guide to the release folder `xemaya-<version>-linux-x86_64.tar.xz`. It is an **alpha**: read
[Known limitations](#known-limitations) before pointing it at a GPU that other work depends on.

## What you need

| | |
| --- | --- |
| CPU | x86-64 with AVX2 |
| GPU | An Intel Arc GPU (or several). Developed on the Arc Pro B70, B65 and B60. |
| System | Linux with glibc 2.38 or later (Ubuntu 24.04 and later, Fedora 40 and later) |
| GPU runtime | Intel's compute runtime with Level Zero. Ubuntu: `sudo apt install intel-opencl-icd libze-intel-gpu1 libze1`. Fedora: `sudo dnf install intel-compute-runtime oneapi-level-zero`. Your user must be in the `render` group. |
| Python | 3.10 or later, with `venv` (Ubuntu: `python3-venv`) |
| A model | A GGUF of one of the three supported families (Qwen3.8-Flash-Next, GLM-5.3-Flash, DeepSeek-V4.1-Flash): see the [model table](README.md#models) |

Intel's SYCL and oneMKL run-time libraries (oneAPI 2026.2) are needed too and are **not** in the folder: `./xemaya.sh setup`
installs them from PyPI into the folder's own `.venv` (Intel's `dpcpp-cpp-rt` and `onemkl-sycl-blas` wheels, under
Intel's licence, about 2.5 GB on disk). Where oneAPI 2026.2 is installed in `/opt/intel/oneapi`, use that instead with
`XEMAYA_ONEAPI=system ./xemaya.sh setup`.

## Install and check

```bash
sha256sum -c SHA256SUMS --ignore-missing
tar -xf xemaya-<version>-linux-x86_64.tar.xz
cd xemaya-<version>
./xemaya.sh setup
./xemaya.sh doctor          # the engine's version and the GPUs it sees
./xemaya.sh selftest        # seconds on the first GPU
```

`./xemaya.sh selftest --cpu` runs the GLM and DeepSeek kernels' parity tests on the OpenCL CPU device instead: no GPU is opened.

## Run a model

```bash
./xemaya.sh models /path/to/your/models      # lists what it found (JSON), writes configs and tokenizers
./xemaya.sh serve <id> --port 8095 --no-api-key
```

`<id>` is a model's `"id"` in the list. Look at its `"ready"`, `"state"` and `"note"`: `verified` is a record of
limited reference checks of that model family, not of your file. The server speaks the OpenAI and Anthropic APIs on
`http://127.0.0.1:8095` and has a web page at the same address. `--host 0.0.0.0 --api-key <secret>` opens it to your
network.

Several GPUs: `ONEAPI_DEVICE_SELECTOR=level_zero:0,1 ./xemaya.sh serve <id> ...` names them (`./xemaya.sh doctor` lists
them in order). The models' layers are split across the cards (`--layer-split auto` is in the generated config). A
card on a slow link (behind the chipset, for example) is left out unless `XEMAYA_SLOW_LINK_GPUS=1` is set.

Configs and tokenizers are written to `~/.local/share/xemaya` (`XEMAYA_HOME`). Nothing is installed outside the
folder and that directory: removing both removes XeMaya.

## Known limitations

- **Alpha.** Reference agreement is recorded for specific files, prompts and GPUs only ([README](README.md#status),
  [STATUS](xemaya/STATUS.md)). Expect rough edges; report them with the file, the settings and the hardware.
- **What this binary was checked on.** The release build passed the host tests and the kernels' parity tests on the
  OpenCL CPU device, and a clean install in a container. On GPUs, each published build runs DeepSeek-V4.1-Flash
  Coder IQ2_XXS on Arc Pro B70 + B65 + B60 against the llama.cpp reference before it is published; the result is in
  that release's notes. GLM-5.3 and Qwen3.8 were not run with the release build: their results in the README were
  measured with development builds of earlier commits, with the same compiler.
- **MiniMax-M3.1-Flash is not in this release.** Its code is in the source and it is held back: slow, and one
  development run ended in a GPU engine reset whose cause is not known.
- A GPU that shares its link with other devices can be starved by long expert streaming; watch `journalctl -k` on the
  first long runs.
- Intel GPUs only in this release. The Windows build with the CUDA target and the hybrid segment (a model's last
  blocks on another machine) are in the source, compile, and have not been run on GPUs: experimental.
- The inherited XeStrata installer (`setup.py`, `xestrata`) and its documents in `docs/` are in the source for
  reference; they download XeStrata's packs and are not this release's install path.
- Intel Graphics Compiler 2.25 to 2.31 miscompiles one kernel pattern on Xe2 and later; the engine turns the affected
  pass off by itself and says so at the start (a few percent of decode speed on short prompts).

## Building it yourself

`tools/xemaya/release.sh` builds this folder from a commit with Intel oneAPI 2026.2 (`icpx`); the
[README](README.md#building) has the direct CMake commands. The source of this exact build is the folder itself
(everything but `engine/`, `xemaya.sh` and `BUILDINFO`), and `xemaya-<version>-source.tar.gz`.

## Licences

XeMaya is under the GNU LGPL, version 3 or later, with code from Strata, Project Maya and ggml / llama.cpp under MIT:
[NOTICE](NOTICE) lists every holder. `engine/lib/` holds oneMath (Apache-2.0), built from
[XeStrata's fork](https://github.com/MistVVK/oneMath). Intel's run-time libraries come from Intel under their own
licences and are not part of XeMaya. Each model has its own licence.
