#!/bin/bash
#PBS -N histogram_eq
#PBS -o output.log
#PBS -e error.log
#PBS -P col7880.ee1221163.course
#PBS -l select=1:ncpus=8:ngpus=1:mem=16G
#PBS -l walltime=00:30:00

module purge
module load compiler/gcc/9.1.0
module load suite/nvidia-hpc-sdk/20.7/cuda11.0

export OMP_NUM_THREADS=8

cd $HOME/histogram_eq/COL380-histogramequalizationCUDA

# ---------------------------------------------------------------------------
#  Build
# ---------------------------------------------------------------------------
echo "========================================"
echo " BUILD"
echo "========================================"
make clean && make
if [ $? -ne 0 ]; then
    echo "BUILD FAILED — aborting"
    exit 1
fi

nvidia-smi

# ---------------------------------------------------------------------------
#  Helper: run one test case
#  seq=1 means also run sequential reference (only for small n)
# ---------------------------------------------------------------------------
run_test() {
    local label=$1
    local n=$2
    local k=$3
    local T=$4
    local run_seq=$5   # 1 = run sequential reference, 0 = skip

    echo ""
    echo "========================================"
    echo " TEST: $label  (n=$n, k=$k, T=$T)"
    echo "========================================"

    python3 gen_input.py $n $k $T input.txt

    ./histogram_eq input.txt
    if [ $? -ne 0 ]; then
        echo "RUNTIME ERROR on $label"
        return
    fi

    # Sanity check + approx MAE vs exact
    python3 - << PYEOF
def verify(tag, infile, outfile):
    with open(infile) as f:
        n = int(f.readline())
        f.readline(); f.readline()
        inp = [f.readline().split() for _ in range(n)]
    with open(outfile) as f:
        out = [l.split() for l in f if l.strip()]
    if len(out) != n:
        print(f"  [{tag}] FAIL: expected {n} lines, got {len(out)}")
        return
    for i, (a, b) in enumerate(zip(inp, out)):
        if a[0] != b[0] or a[1] != b[1] or a[2] != b[2]:
            print(f"  [{tag}] FAIL: coords changed at line {i}")
            return
        v = int(b[3])
        if not (0 <= v <= 255):
            print(f"  [{tag}] FAIL: intensity {v} out of range at line {i}")
            return
    print(f"  [{tag}] OK")

verify("knn",        "input.txt", "knn.txt")
verify("approx_knn", "input.txt", "approx_knn.txt")
verify("kmeans",     "input.txt", "kmeans.txt")

with open("knn.txt") as f:
    knn = [int(l.split()[3]) for l in f]
with open("approx_knn.txt") as f:
    approx = [int(l.split()[3]) for l in f]
mae = sum(abs(a-b) for a,b in zip(knn,approx)) / len(knn)
print(f"  [approx_knn vs knn] MAE = {mae:.4f}")
PYEOF

    # Sequential reference (only for small n — O(N^2) is too slow for large)
    if [ "$run_seq" = "1" ]; then
        echo "  Running sequential reference..."
        python3 sequential.py input.txt
    fi
}

# ---------------------------------------------------------------------------
#  Test suite
# ---------------------------------------------------------------------------
# edge cases — with sequential reference
run_test "edge_k1"   1000    1  10   1
run_test "small"     1000   10  20   1
run_test "medium"    10000  32  20   1

# large cases — skip sequential (too slow), just verify format + MAE vs exact
run_test "large"     100000  32  20  0
run_test "max"       100000 128  50  0

echo ""
echo "========================================"
echo " ALL TESTS DONE"
echo "========================================"

