# Real Time Quantized CNN Accelerator for Edge AI

**Final project in Computer Engineering - Project 335**
Faculty of Engineering, Bar-Ilan University

Authors: Roni Volshtein, Kinanah Hanif  
Academic supervisor: Prof. Leonid Yavits  
Project mentor: David Freud  
Track: Hardware Design

## What this project does

This project builds a hardware accelerator for quantized CNN inference on an
AMD/Xilinx **PYNQ-Z1** FPGA, using the **FINN** framework.

A pretrained **CNV-w1a1** model - a VGG style CNN with 6 convolutional and
3 fully connected layers, trained on CIFAR-10 with 1 bit weights and
activations - is compiled into a **streaming dataflow** hardware architecture.
Every network layer becomes its own hardware block, and activations stream
between blocks through on chip FIFOs instead of going out to DRAM.

We then ran a **design space exploration** over the folding parameters
(PE and SIMD) across four configurations, A through D, repeatedly locating the
slowest pipeline stage and widening it.

## Scope 

All performance results in this repository come from **cycle accurate RTL
simulation (PyVerilator RTLSIM)** and **Vivado Out of Context synthesis**.

Measurement of a single thread CPU baseline is compared with simulated FPGA performance. 

Power figures are estimates. They come from Vivado's
`report_power` on the placed and routed out of context design - the accelerator
alone, without the Zynq processing system. See [Power and energy](#power-and-energy).

---

## Results

### Design space exploration (A → D)

| Config | Change made | Bottleneck | RTLSIM FPS | Stable FPS | Latency (ms) | LUT | FF | BRAM | DSP | Fmax (MHz) | WNS (ns) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A (baseline) | default FINN folding | MVAU_hls_6 | 1725.66 | 2254.64 | 1.3596 | 20,466 | 27,932 | 98 | 0 | 104.83 | +0.461 |
| B | MVAU_hls_6 SIMD 4 → 8 | MVAU_hls_7 | 2324.99 | 3220.61 | 1.1961 | 20,499 | 27,983 | 98 | 0 | 109.77 | +0.890 |
| **C (recommended)** | MVAU_hls_7 SIMD 8 → 16 | MVAU_hls_0 | **2416.88** | 3220.61 | **1.0326** | 20,607 | 28,014 | 98 | 0 | 103.52 | **+0.340** |
| D (maximum) | MVAU_hls_0 PE 16 → 32 | MVAU_hls_3 | 2488.68 | 3329.47 | 1.0147 | 21,886 | 29,122 | 99 | 0 | 100.83 | +0.082 |

Latency is `latency_cycles` at a 10 ns clock period (100 MHz).
Source data: [`results/finn_A_D_final_results.csv`](results/finn_A_D_final_results.csv)

The bottleneck column names the MVAU unit that FINN reported with the
highest `max_cycles`. All four configurations contain the same nine MVAU units,
`MVAU_hls_0` through `MVAU_hls_8` - six convolutional and three fully connected.
Folding changes how much parallelism each unit is given.  
What moves between configurations is which unit is the limit.

**Configuration C is the recommended design.** D is marginally faster (+3.0%
RTLSIM throughput) but costs 1,279 more LUTs and 1,108 more FFs, and drops the
timing margin from +0.340 ns to +0.082 ns. C keeps roughly four times the slack
for almost the same performance.

**Zero DSP slices are used in any configuration.** With 1 bit weights and
activations the multiply accumulate operation is implemented using XNOR and popcount operations, which
maps onto LUTs rather than DSP blocks.

### Power and energy

| Config | Total on chip (W) | Dynamic (W) | Static (W) | Energy / inference (mJ) | FPS per Watt |
|---|---|---|---|---|---|
| A (baseline) | 0.660 | 0.543 | 0.118 | 0.382 | 2,615 |
| B | 0.660 | 0.542 | 0.118 | 0.284 | 3,523 |
| **C (recommended)** | **0.631** | 0.514 | 0.117 | **0.261** | **3,830** |
| D (maximum) | 0.681 | 0.562 | 0.118 | 0.274 | 3,654 |

Configuration C draws the least power of the four and delivers the lowest energy
per inference - 31.7% below the baseline. D buys about three percent more
throughput for 7.9% more power and ends up less energy efficient than C. The
recommendation of C, argued above on timing margin and resource cost, holds
independently on the energy axis.

For C the breakdown is 0.165 W in Block RAM, 0.149 W in signals, 0.123 W in
slice logic, 0.078 W in clocking and 0.117 W of device static power. Junction
temperature is 32.3 °C at a 25 °C ambient, against a maximum permissible ambient
of 77.7 °C.

The synthesis is **out of context**: the design analysed is the `finn_design_wrapper` accelerator alone, without the Zynq processing system,
the DMA engines or the PYNQ shell, so these are accelerator figures and not board
figures. **No switching activity file was supplied**, so Vivado propagated
default activity rates and every report states a confidence level of **Medium**.
And energy per inference combines an estimated power with a simulated
throughput, so it carries the same caveat as the speedup ratios.

These numbers were not produced by a separate analysis. FINN's
`step_out_of_context_synthesis` runs `deps/oh-my-xilinx/vivadocompile.tcl`, which
ends with `open_run impl_1` followed by `report_utilization`, `report_timing` and
`report_power` in one Vivado session - so the power figures come from the same
run as the LUT, FF, BRAM and WNS figures in the table above.

Full reports: [`results/vivado_power_reports.txt`](results/vivado_power_reports.txt)
Summary data: [`results/power_A_D.csv`](results/power_A_D.csv)

### CPU baseline vs. Configuration C

| Platform | Method | Throughput | Latency |
|---|---|---|---|
| Host CPU | PyTorch, 1 thread, batch = 1 | 42.63 ± 1.65 FPS | 23.49 ± 0.93 ms |
| FINN Config C | RTLSIM - 100 MHz | 2416.88 FPS | 1.033 ms |
| Ratio | measured CPU vs. simulated FPGA | ≈ 56.7× | ≈ 22.7× |

CPU baseline: 5 runs × 500 inferences, batch size 1, after a 30 iteration
warm up, timed with `time.perf_counter`.
Source data: [`results/cpu_vs_finn_C.csv`](results/cpu_vs_finn_C.csv)

### Functional verification

Configuration C was verified with `STITCHED_IP_RTLSIM` against the PyTorch
software model. Both returned class 3 - `Match = True`. This is a single input
consistency check.

---

## Running this yourself

The notebooks under `notebooks/`. They are not standalone scripts: they
import `finn.*` and `qonnx.*` and they invoke Vivado and Vitis HLS, so they only
run **inside the FINN Docker container**. Opening them in a plain Jupyter
install will fail.

### What you need

| Requirement | Notes |
|---|---|
| Linux host with `bash` | WSL2 on Windows also works |
| Docker, usable without `sudo` | This project was developed and tested using FINN’s docker based workflow |
| Vivado + Vitis HLS 2022.2 | Installed on the host, not inside the container |
| ~8 GB RAM | Minimum for Zynq class targets |
| Tens of GB of free disk | FINN build directories are large |

No GPU is required - the container we used had none, and everything here runs on
CPU. No PYNQ-Z1 board is required either. A physical board is required only for hardware deployment.

### What you do *not* need to install

You do **not** need to install Brevitas, QONNX, finn experimental, PyVerilator or
any other FINN dependency yourself. FINN fetches them: `fetch-repos.sh` clones
each one into `deps/` inside your FINN checkout, and they are pip installed in
editable mode when the container starts. You will see it in the startup log,
roughly like this:

```
Obtaining file:///home/<user>/workspace/finn/deps/brevitas
...
Installing collected packages: unfoldNd, brevitas
  Running setup.py develop for brevitas
Successfully installed brevitas
```

That is FINN installing its own pinned dependency, not something to do by hand.

### Steps

```bash
# 1. Point FINN at your Xilinx installation
export FINN_XILINX_PATH=/tools/Xilinx      # your install root
export FINN_XILINX_VERSION=2022.2

# 2. Get FINN, at the version this project used
git clone https://github.com/Xilinx/finn
cd finn
git checkout v0.10.1

# 3. Start the notebook server inside the container
bash ./run-docker.sh notebook
```

The first launch is slow: it pulls the image and clones the dependencies into
`deps/`. Jupyter then starts on port 8888 and prints a token URL in the terminal.
Copy this repository's `notebooks/` folder into the FINN working directory so
Jupyter can see it, then run the notebooks in the order in the table below.

### What to expect

Hardware generation - `step_hw_codegen`, `step_hw_ipgen` and the Vivado
synthesis that follows - is the slow part. The notebooks themselves note it takes
roughly **30 minutes per configuration**, depending on the host, and RTL
simulation of the stitched design adds more. Budget several hours to reproduce
all four configurations from scratch.

### What you will have to edit

Four notebooks read their report files from a path hardcoded to the machine they
were run on:

```python
report_dir = "/tmp/finn_dev_kinanah/cnv_folding_C/folding_C_performance/report"
```

It appears in `cnv_folding_optimization.ipynb` (cell 34), `cnv_folding_C.ipynb`
(cell 35), `cnv_folding_D.ipynb` (cell 35) and `cnv_final_verification.ipynb`
(cell 28). Replace `/tmp/finn_dev_kinanah` with your own `FINN_BUILD_DIR` before
running those cells. Every other path in the notebooks is derived from
`os.environ["FINN_BUILD_DIR"]` and needs no change.

Exact figures depend on the FINN version, so a different release may produce
different cycle counts and resource numbers.

---

## Repository layout

```
notebooks/   Jupyter notebooks, with all outputs preserved
results/     Measurement data (CSV), the Vivado power reports, and the charts used in the book
hardware/    FINN generated deployment packages and Vivado block diagrams
configs/     The folding parameters for configurations A–D
```

Appendix A of the project book lists thirteen notebooks in total, eight are here. The rest
are the official FINN tutorial notebooks - `0_how_to_work_with_onnx`,
`1_brevitas_network_import_via_QONNX`, the cybersecurity MLP series and the FINN
end to end examples - which we worked through while learning the framework. They
belong to the FINN project.  
The notebooks kept here are the ones this project ran: they derive from the FINN end to end example
flow and were extended with the per layer folding configurations, the
bottleneck driven exploration procedure, the `STITCHED_IP_RTLSIM` verification
harness and the CPU baseline benchmark.

### Which notebook produces what

| Notebook | What it produces |
|---|---|
| `cnv_end2end_baseline.ipynb` | Configuration A - first full hardware build, identifies MVAU_hls_6 as the bottleneck |
| `cnv_folding_optimization.ipynb` | Configuration B - first SIMD widening |
| `cnv_folding_C.ipynb` | Configuration C - the recommended design |
| `cnv_folding_D.ipynb` | Configuration D - maximum throughput |
| `cnv_final_verification.ipynb` | `STITCHED_IP_RTLSIM` functional verification and the CPU baseline benchmark |
| `cnv_cpu_gpu_benchmark.ipynb` | The environment check recording why there is no GPU baseline, plus an independent reproduction of the Configuration A reports |
| `tfc_end2end_baseline.ipynb` | Fully connected (TFC) reference flow |
| `tfc_end2end_verification_B.ipynb` | Verification of the TFC flow - `cppsim` and node by node RTL simulation were exercised here, on the TFC network only |

Each of the four configuration notebooks also carries its Vivado power report in
the output of its build cell; those reports are collected in
`results/vivado_power_reports.txt`.

Note on the CPU benchmark. `cnv_final_verification.ipynb` holds more than
one benchmark cell: the figure used throughout the project book is the **5 run
average, 42.63 FPS and 23.49 ms** (cell 35), while an earlier single run cell
reporting 46.28 FPS is kept for transparency.

Not every line of text in the notebooks is ours. These notebooks were derived
from the official FINN end to end tutorials, and the tutorials' explanatory
markdown was kept alongside our own work rather than stripped out. That text
describes the tutorial's model, not our measurements.

---

## Environment

| Component | Version |
|---|---|
| FINN | v0.10.1 |
| Vivado / Vitis HLS | 2022.2 |
| Python | 3.10 |
| PyTorch | 1.13.1+cu116 |
| Brevitas | v0.10.0, pinned by FINN and installed from deps/brevitas |
| Target board | AMD/Xilinx PYNQ-Z1 (XC7Z020-CLG400) |
| Clock target | 100 MHz (10 ns) |

FINN was run from its official Docker container, with Vivado mounted in from the
host. Brevitas was not installed independently - FINN pins it and installs it
from `deps/brevitas`.

---

## Notes

Figures taken from the official FINN documentation are used in the project book
but are deliberately **not** redistributed here.

The full write up is in the project book, *Real Time Quantized CNN Accelerator
for Edge AI*, Project 335, Bar-Ilan University.
