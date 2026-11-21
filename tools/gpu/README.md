# GPU gate

Checks a build with a GPU back end against the CPU on a machine with that
GPU (Phase 11, 12; `docs/release.md`, *GPU hosts*): `BACKEND=hip`
(default) or `BACKEND=cuda`. For each input and case it runs `sesmesh`
four times: CPU, `--backend $BACKEND` (exact) and twice with
`--refine fast`. It then checks that:

* every run gives the same exit code;
* exact mode writes the CPU's OBJ and statistics, byte for byte (timings,
  back end and device aside);
* fast mode writes the CPU's face lines, and the same OBJ when repeated;
* fast mode's area and volume are within 10⁻⁶ (relative) of the CPU's.

```bash
export PATH=$HOME/.cargo/bin:/opt/rocm/bin:$PATH
cargo +stable build --release -p sesmesh-cli --features hip
cargo xtask fetch --set small --set bench --set validation --set sweep
ls tests/data/cache > /tmp/gate-inputs.txt
tools/gpu/gate.sh /tmp/gate /tmp/gate-inputs.txt                 # SES at 1, 0.5, 0.25 Å
tools/gpu/gate.sh /tmp/gate-sv /tmp/gate-inputs.txt sas:1 sas:0.5 vdw:1 vdw:0.5   # SAS, vdW
```

The last line of the output gives the number of differences; the exit code
is non-zero when there are any. The CSV rows are in `OUT_DIR/gate-*.csv`
(columns in `gate.sh`). On a 16-core Zen 5 with a Radeon 8060S the SES gate
over the 256 inputs of those sets (768 cases) takes about 8.5 minutes, the
SAS and vdW gate over the 53 inputs of the validation set alone (212 cases)
1.5 minutes.

With `BACKEND=cuda` on a laptop (RTX 2070 Max-Q, 6-core i7) and the CPU
references made on the Zen 5, the SES gate's GPU side took 91
minutes, most of it 3J3Q at 0.25 Å: 800 M triangles, a 72 GB OBJ per run,
more than 62 GB of memory (it swapped). Leave it out of the list on
smaller machines.

`gate.sh` needs bash, GNU coreutils (`split`, `sha256sum`), `grep` and
`jq`; it uses no Python. `summary.sh GATE_CSV...` prints the totals of
existing CSV files again (for example after merging runs). `BIN`, `DATA` and `JOBS` override the
binary, the input directory and the number of parallel shards; `SIMD`
passes `--simd` to every run.

**CPU references from another machine.** When the GPU machine's CPU is
slow, run the CPU side elsewhere with the same binary and inputs, at the
same absolute paths (the statistics record the input path):

```bash
MODE=cpu SIMD=avx2 tools/gpu/gate.sh /tmp/ref /tmp/gate-inputs.txt          # fast CPU
# copy /tmp/ref/ref to the GPU machine, then there:
MODE=gpu REF=/tmp/ref/ref BACKEND=cuda SIMD=avx2 JOBS=2 tools/gpu/gate.sh /tmp/gate /tmp/gate-inputs.txt
```

`MODE=cpu` writes one record per case (exit code; SHA-256 of the OBJ and
of its face lines; the statistics), a few kilobytes instead of the OBJ.
CPU output does not depend on the SIMD level, but the statistics name it:
set `SIMD` to a level both CPUs support.
