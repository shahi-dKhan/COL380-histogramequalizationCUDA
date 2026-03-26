#!/bin/bash
#PBS -N histogram_eq_compare
#PBS -o output_v2.log
#PBS -e error_v2.log
#PBS -P col7880.ee1221163.course
#PBS -l select=1:ncpus=8:ngpus=1:mem=16G
#PBS -l walltime=01:00:00

module purge
module load compiler/gcc/9.1.0
module load suite/nvidia-hpc-sdk/20.7/cuda11.0

export OMP_NUM_THREADS=8

cd $HOME/histogram_eq/COL380-histogramequalizationCUDA

echo "========================================"
echo " BUILD"
echo "========================================"
make clean && make
if [ $? -ne 0 ]; then
    echo "BUILD FAILED"
    exit 1
fi

nvidia-smi

run_test() {
    local label=$1
    local n=$2
    local k=$3
    local T=$4
    local run_seq=$5

    echo ""
    echo "========================================"
    echo " TEST: $label  (n=$n, k=$k, T=$T)"
    echo "========================================"

    python3 gen_input.py $n $k $T input.txt

    echo "--- v1 ---"
    ./histogram_eq input.txt
    if [ $? -ne 0 ]; then echo "v1 RUNTIME ERROR on $label"; return; fi

    echo "--- v2 ---"
    ./histogram_eq_v2 input.txt
    if [ $? -ne 0 ]; then echo "v2 RUNTIME ERROR on $label"; return; fi

    # Python inline: verify output format for both, compute MAE between v1 and v2
    python3 - << 'PYEOF'
import sys, os

label = os.environ.get('TEST_LABEL', '')

def verify(tag, infile, outfile):
    with open(infile) as f:
        n = int(f.readline())
        f.readline(); f.readline()
        inp = [f.readline().split() for _ in range(n)]
    with open(outfile) as f:
        out = [l.split() for l in f if l.strip()]
    if len(out) != n:
        print(f"  [{tag}] FAIL: expected {n} lines, got {len(out)}")
        return False
    for i, (a, b) in enumerate(zip(inp, out)):
        if a[0] != b[0] or a[1] != b[1] or a[2] != b[2]:
            print(f"  [{tag}] FAIL: coords changed at line {i}")
            return False
        v = int(b[3])
        if not (0 <= v <= 255):
            print(f"  [{tag}] FAIL: intensity {v} out of range at line {i}")
            return False
    print(f"  [{tag}] format OK")
    return True

def mae(f1, f2):
    a = [int(l.split()[3]) for l in open(f1)]
    b = [int(l.split()[3]) for l in open(f2)]
    return sum(abs(x-y) for x,y in zip(a,b)) / len(a)

# Verify all outputs
verify("v1 knn",        "input.txt", "knn.txt")
verify("v1 approx_knn", "input.txt", "approx_knn.txt")
verify("v1 kmeans",     "input.txt", "kmeans.txt")
verify("v2 knn",        "input.txt", "knn_v2.txt")
verify("v2 approx_knn", "input.txt", "approx_knn_v2.txt")
verify("v2 kmeans",     "input.txt", "kmeans_v2.txt")

# MAE between v1 and v2 (should be 0 for knn and kmeans, small for approx)
print(f"  [knn      v1 vs v2] MAE = {mae('knn.txt',        'knn_v2.txt'):.4f}  (expect 0)")
print(f"  [approx   v1 vs v2] MAE = {mae('approx_knn.txt', 'approx_knn_v2.txt'):.4f}  (approx may differ)")
print(f"  [kmeans   v1 vs v2] MAE = {mae('kmeans.txt',     'kmeans_v2.txt'):.4f}  (expect 0)")

# approx vs exact for both versions
knn   = [int(l.split()[3]) for l in open('knn.txt')]
apx1  = [int(l.split()[3]) for l in open('approx_knn.txt')]
apx2  = [int(l.split()[3]) for l in open('approx_knn_v2.txt')]
n = len(knn)
print(f"  [v1 approx vs exact] MAE = {sum(abs(a-b) for a,b in zip(apx1,knn))/n:.4f}")
print(f"  [v2 approx vs exact] MAE = {sum(abs(a-b) for a,b in zip(apx2,knn))/n:.4f}")
PYEOF

    if [ "$run_seq" = "1" ]; then
        echo "  Running sequential reference..."
        python3 sequential.py input.txt
    fi
}

# edge cases
run_test "edge_k1"   1000    1  10   1
run_test "small"     1000   10  20   1
run_test "medium"    10000  32  20   1

# large cases
run_test "large"     100000  32  20  0
run_test "max"       100000 128  50  0

echo ""
echo "========================================"
echo " ALL TESTS DONE"
echo "========================================"
