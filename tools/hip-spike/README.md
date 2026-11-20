# HIP feasibility spike

Measurements behind `docs/05-gpu-hip.md` §2 (issue #24). Not part of the
workspace and not a back end: hand-written `extern "C"` bindings to
`libamdhip64`, code objects built by `hipcc`, and two programs in one binary.

Needs ROCm (tested with ROCm 7.16 on gfx1151) and Rust ≥ 1.89.

```bash
cd tools/hip-spike
SPIKE_ARCH=gfx1151 cargo build --release     # ROCM_PATH defaults to /opt/rocm
./target/release/hip-spike                   # arithmetic agreement, rates, latency, bandwidth
```

`spike.hip` holds the micro-benchmarks. `ses.hip` is a line-by-line port of
`field::kernel::ses_kernel` (plus an f64-compute variant) that runs on
sub-blocks dumped by a patched sesmesh and compares with the CPU values bit
for bit:

```bash
git apply tools/hip-spike/profile-and-dump.patch      # temporary counters + dump
cargo build --release -p sesmesh-cli
SESMESH_DUMP=4hhb.bin target/release/sesmesh 4HHB.cif.gz -s 0.25 -j 1 -o /dev/null
(cd tools/hip-spike && ./target/release/hip-spike ses ../../4hhb.bin)
git apply -R tools/hip-spike/profile-and-dump.patch
```

The patch also prints `PROF <counter> <value>` lines after a `surface` run:
nanoseconds per part of the mesh stage summed over threads (`subclass`,
`gather`, `kernel`, `lgather`, `vertex`, `cells`, `edges_loop`) and candidate
list totals. Dumps are large (4HHB at 0.25 Å: 236 MB; 1AON at 0.5 Å: 1.5 GB).
