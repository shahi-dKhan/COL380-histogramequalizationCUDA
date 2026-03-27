#!/bin/bash
#PBS -N a2_correctness
#PBS -o test_output.log
#PBS -e test_error.log
#PBS -P col7880.ee1221163.course
#PBS -l select=1:ncpus=8:ngpus=1:mem=16G
#PBS -l walltime=01:30:00

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
if [ $? -ne 0 ]; then echo "BUILD FAILED"; exit 1; fi
echo ""

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
PASS=0
FAIL=0

# Compare CUDA output vs sequential reference (exact match, MAE must be 0)
check_exact() {
    # $1=label  $2=cuda_out  $3=ref_out
    result=$(python3 - "$2" "$3" <<'PYEOF'
import sys
def read(f):
    return [int(l.split()[3]) for l in open(f) if len(l.split()) == 4]
a, b = read(sys.argv[1]), read(sys.argv[2])
if len(a) != len(b):
    print(f"FAIL  length mismatch {len(a)} vs {len(b)}")
    sys.exit(0)
errs = [abs(x-y) for x,y in zip(a,b)]
mae  = sum(errs)/len(errs)
mis  = sum(1 for e in errs if e > 0)
if mae == 0.0:
    print(f"PASS  MAE=0.0000  mismatches=0/{len(a)}")
else:
    print(f"FAIL  MAE={mae:.4f}  mismatches={mis}/{len(a)}")
PYEOF
)
    status=$(echo "$result" | cut -d' ' -f1)
    printf "    %-24s  %s\n" "$1" "$result"
    if [ "$status" = "PASS" ]; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
}

# Validity check for large N (sequential too slow): check range, line count, coords
check_valid() {
    # $1=label  $2=cuda_out  $3=input  $4=expected_n
    result=$(python3 - "$2" "$3" "$4" <<'PYEOF'
import sys
outf, inf, expected_n = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    out_lines = [l.split() for l in open(outf) if len(l.split()) == 4]
    in_lines  = [l.split() for l in open(inf)  if len(l.split()) == 4]
    if len(out_lines) != expected_n:
        print(f"FAIL  wrong line count {len(out_lines)} vs {expected_n}")
        sys.exit(0)
    bad_range = sum(1 for p in out_lines if not (0 <= int(p[3]) <= 255))
    bad_coord = sum(1 for a,b in zip(out_lines, in_lines)
                    if a[0]!=b[0] or a[1]!=b[1] or a[2]!=b[2])
    if bad_range > 0 or bad_coord > 0:
        print(f"FAIL  bad_intensity={bad_range}  bad_coords={bad_coord}")
    else:
        print(f"PASS  n={expected_n}  all intensities in [0,255]  coords match")
except Exception as e:
    print(f"FAIL  {e}")
PYEOF
)
    status=$(echo "$result" | cut -d' ' -f1)
    printf "    %-24s  %s\n" "$1" "$result"
    if [ "$status" = "PASS" ]; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
}

# Run one test case
# $1=n  $2=k  $3=T  $4=dist  $5=seed  $6=use_seq (1=compare vs sequential, 0=validity only)
run_case() {
    local n=$1 k=$2 T=$3 dist=$4 seed=${5:-42} use_seq=${6:-1}
    echo "  [n=$n k=$k T=$T dist=$dist]"

    python3 gen_input.py $n $k $T input.txt --dist $dist --seed $seed --range 10000

    ./a2 input.txt knn        > /dev/null 2>&1
    ./a2 input.txt approx_knn > /dev/null 2>&1
    ./a2 input.txt kmeans     > /dev/null 2>&1

    if [ "$use_seq" -eq 1 ]; then
        python3 sequential.py input.txt > /dev/null 2>&1
        check_exact "knn"    knn.txt    knn_seq.txt
        check_exact "kmeans" kmeans.txt kmeans_seq.txt
    else
        check_valid "knn"    knn.txt    input.txt $n
        check_valid "kmeans" kmeans.txt input.txt $n
    fi
    # approx_knn: always validity check (it's an approximation)
    check_valid "approx_knn" approx_knn.txt input.txt $n
    echo ""
}

# ---------------------------------------------------------------------------
#  Test cases
# ---------------------------------------------------------------------------
echo "========================================"
echo " CORRECTNESS TESTS  (vs sequential.py)"
echo " N <= 2000: exact MAE=0 required"
echo " N >  2000: validity check (range + coords)"
echo "========================================"
echo ""

echo "=== Edge / tiny N ==="
run_case    1   0  1 uniform 42 1
run_case    2   1  1 uniform 42 1
run_case   10   3  5 uniform 42 1
run_case   50  10 10 uniform 42 1
echo ""

echo "=== Small N: vary k (n=100, T=20, uniform) ==="
run_case  100   1 20 uniform 42 1
run_case  100   4 20 uniform 42 1
run_case  100  16 20 uniform 42 1
run_case  100  32 20 uniform 42 1
run_case  100  64 20 uniform 42 1
run_case  100  99 20 uniform 42 1
echo ""

echo "=== N=500: vary distribution ==="
run_case  500  32 20 uniform   42 1
run_case  500  32 20 dense     42 1
run_case  500  32 20 clustered 42 1
run_case  500  32 20 skewed    42 1
echo ""

echo "=== N=1000: vary k ==="
run_case 1000   1 20 uniform 42 1
run_case 1000  16 20 uniform 42 1
run_case 1000  32 20 uniform 42 1
run_case 1000  64 20 uniform 42 1
run_case 1000 128 20 uniform 42 1
echo ""

echo "=== N=1000: vary T ==="
run_case 1000  32  1 uniform 42 1
run_case 1000  32  5 uniform 42 1
run_case 1000  32 50 uniform 42 1
echo ""

echo "=== N=2000: distributions ==="
run_case 2000  32 20 uniform   42 1
run_case 2000  32 20 dense     42 1
run_case 2000  32 20 clustered 42 1
run_case 2000  32 20 skewed    42 1
echo ""

echo "=== Large N: validity only ==="
run_case  5000  32 20 uniform   42 0
run_case  5000 128 20 uniform   42 0
run_case 10000  32 20 uniform   42 0
run_case 10000  32 20 dense     42 0
run_case 10000  32 20 clustered 42 0
run_case 10000  32 20 skewed    42 0
run_case 10000   1 20 uniform   42 0
run_case 10000 128 20 uniform   42 0
run_case 25000  32 20 uniform   42 0
run_case 50000  32 20 uniform   42 0
run_case 50000  64 30 uniform   42 0
run_case 100000 32 20 uniform   42 0
run_case 100000 128 50 uniform  42 0
echo ""

# ---------------------------------------------------------------------------
#  Summary
# ---------------------------------------------------------------------------
TOTAL=$((PASS + FAIL))
echo "========================================"
printf " SUMMARY:  %d / %d passed\n" $PASS $TOTAL
if [ $FAIL -eq 0 ]; then
    echo " ALL TESTS PASSED"
else
    echo " $FAIL TEST(S) FAILED"
fi
echo "========================================"
