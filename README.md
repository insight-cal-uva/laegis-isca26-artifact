# LÆGIS: Pinpointing and Addressing Performance Overheads of GPU-based Confidential Computing

This repository contains the artifact for our ISCA 2026 paper on **LÆGIS**. It includes the paper PDF, the open-source components needed to study our setup, and supporting code/data used for evaluation and figure generation.

At a high level, this artifact includes:

- An instrumented NVIDIA kernel driver module tree in `nvidia-550.163.01`, with a built-in reproduction guide in `nvidia-550.163.01/HOWTO.txt`. This component is used to reproduce the paper results corresponding to Figure 3, Figure 4, Listing 1, and Figure 15 (UVM).
- A modified GPGPU-Sim based simulator in `gpgpu-sim-uvm` with added Unified Virtual Memory (UVM) support. This component is used for the simulation-based results in Figure 8, 12, 13, 14, 16.
- A `benchmark` directory containing applications we used for main results.
- The camera-ready paper PDF can be accessed [[here](https://adwaitjog.github.io/docs/pdf/yang_laegis_isca26.pdf)].
## Included vs. Omitted Components

The LÆGIS-specific implementation details used for Figure 12, Figure 13, Figure 14, and Figure 16 are **not included** in this public artifact. These mechanisms were implemented on top of the simulator infrastructure included in this repository, but the LÆGIS-specific code paths themselves are currently withheld for now and will be updated in the future.

If you have questions about those implementation details, please contact the author at `yangyang@virginia.edu` and we can discuss them further.

## Licensing

Unless otherwise specified, the contents of this repository are covered by the top-level repository license. Some subdirectories and integrated third-party components carry their own licenses or upstream licensing terms, which take precedence for those parts.

- `benchmark` includes its own license file at `benchmark/LICENSE`.
- Some benchmark subcomponents also carry their own license files.
- `gpgpu-sim-uvm` is based on upstream GPGPU-Sim and related integrated components; please consult the documentation and notices shipped in that directory before reuse.
- `nvidia-550.163.01` contains an instrumented driver tree; please follow the applicable licensing terms and notices associated with that codebase.
- Scripts and auxiliary files may inherit the licensing terms of the component they accompany unless otherwise noted.

## Citation

If you use this instrumented driver module, `gpgpu-sim-v4.2` with UVM, or LÆGIS design, please cite our paper:

```bibtex
@inproceedings{yang2026laegis,
  title     = {{LÆGIS: Pinpointing and Addressing Performance Overheads of GPU-based Confidential Computing}},
  author    = {Yang, Yang and Jog, Adwait},
  booktitle = {Proceedings of the IEEE/ACM International Symposium on Computer Architecture (ISCA)},
  year      = {2026}
}
```