<h1 align="center">
GPGPU-Sim Framework
</h1>
<p align="center">
YANG YANG
</p>

<p align="center">
This repository contains the codebase and documentation of the GPGPU-Sim simulator, which aims to reproduce GPU related simulation works on the University of Virginia's servers.

This GPGPU-Sim simulator was initially developed based on [GPGPU-Sim version 4.0.0](https://github.com/gpgpu-sim/gpgpu-sim_distribution). We modified the simulator to support Unified Virtual Memory (UVM) by integrating [UVMSmart](https://github.com/DebashisGanguly/gpgpu-sim_UVMSmart) into it. Later, we updated it to [GPGPU-Sim version 4.2.0](https://github.com/gpgpu-sim/gpgpu-sim_distribution/tree/dev) with the AccelWattch power model enabled.

Now, we have replace the DRAM model with [Ramulator](https://github.com/CMU-SAFARI/GPGPUSim-Ramulator). Currently, HBM and GDDR5 have been tested. Other memory types will be tested in the future. 
</P>

> Note: the orginal README file of GPGPU-Sim could be found [here](./ORI-README.md).