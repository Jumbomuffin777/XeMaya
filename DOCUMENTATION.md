<!--
SPDX-FileCopyrightText: 2026 the XeMaya contributors
SPDX-License-Identifier: LGPL-3.0-or-later
-->
# XeMaya documentation

Start with the [README](README.md) for the current model matrix, build entry point and limitations.
This documentation was audited against source snapshot `32c979a0a6de0c538e524ee4e8092577157f3a93`.

| Need | Read | Scope |
| --- | --- | --- |
| Current qualification and incidents | [Status](xemaya/STATUS.md) | Latest handoff first; older dated sections are historical, even when their headings say “unverified” or “not started”. |
| Architecture and remaining work | [Plan](xemaya/PLAN.md) | Plans are not implemented capabilities. |
| Changes in this documentation update | [Changelog](CHANGELOG.md) | Evidence links and validation limits. |
| Local model discovery | [catalog.py](tools/xemaya/catalog.py) | Three released architecture families; JSON capabilities, prerequisite checks and optional config/tokenizer writes. |
| Installing a release | [INSTALL.md](INSTALL.md) | The Linux release folder for Intel GPUs: requirements, `xemaya.sh`, known limitations. |
| Direct oneAPI build | [README build instructions](README.md#building) | XeMaya CMake accepts icpx for Intel GPUs; setup still follows XeStrata's compiler policy. |
| Qwen installation / packaging | [XeStrata guide](docs/XESTRATA.md), [Build](docs/BUILD.md) | Inherited workflow, command and package names. Upstream download links point to XeStrata. |
| Server / API options | [Details](docs/DETAILS.md#using-it) | Inherited server features; availability for a particular model needs separate qualification. |
| Intel kernels and device behaviour | [Xe](docs/XE.md), [Multi-GPU](docs/MULTIGPU.md) | Inherited architecture notes and measurements. |
| Image, batching and other inherited paths | [Images](docs/IMAGE.md), [Batching](docs/BATCHING.md), [ORCA](docs/ORCA.md), [KVA](docs/KVA.md) | Source/docs are inherited; inclusion does not establish fresh XeMaya validation. |
| Contributing and licensing | [Contributing](CONTRIBUTING.md), [NOTICE](NOTICE), [Licences](LICENSES/) | Preserve upstream attribution and per-file licences. |

## Reading evidence correctly

- The product name is **XeMaya**, one word. XeStrata and Strata identify upstream projects.
- `strata`, `xestrata`, `STRATA_*`, `XESTRATA_*` and `XEMAYA_*` are existing executable/configuration
  identifiers; use them as implemented. A documentation rename does not rename these interfaces.
- Catalog `verified` is family-level, based on specific reference comparisons. `ready` checks local
  prerequisites. Neither proves that every quantisation, context length or API feature works.
- The catalog currently reports no vision capability for any of its families. Inherited image
  documentation does not mean vision or DSpark drafting has been implemented. MiniMax-M3.1-Flash is held back
  from the releases; the catalog lists it only with `XEMAYA_UNRELEASED=1`.
- Numbers in README and STATUS are existing recorded results, not fresh benchmarks. Read the file,
  prompt, GPU split, cold/warm state and date before comparing them.
- Old operational notes are history, not permission to run a job or change a service. The latest
  handoff takes precedence when reading status, and records the current stop after MiniMax's reset.

## Support

[Buy Me a Coffee — Jumbomuffin777](https://buymeacoffee.com/Jumbomuffin777)
