#!/bin/bash
#PBS -N histogram_eq
#PBS -o output.log
#PBS -e error.log
#PBS -q standard
#PBS -P col7880.ee1221163.course
#PBS -l select=1:ncpus=8:ngpus=1:mem=16G
#PBS -l walltime=04:00:00

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

nvidia-smi
echo ""

# ---------------------------------------------------------------------------
#  CSV header
# ---------------------------------------------------------------------------
RESULTS="results.csv"
echo "group,label,n,k,T,dist,range,knn_s,approx_s,kmeans_s,approx_mae_vs_knn,knn_mae_vs_seq,kmeans_mae_vs_seq" > $RESULTS

# ---------------------------------------------------------------------------
#  run_case  group label n k T dist [run_seq=0] [range=10000]
# ---------------------------------------------------------------------------
run_case() {
    local group=$1 label=$2 n=$3 k=$4 T=$5 dist=$6 run_seq=${7:-0} R=${8:-10000}

    echo "[$group] $label  n=$n k=$k T=$T dist=$dist R=$R"

    python3 gen_input.py $n $k $T input.txt --dist $dist --seed 42 --range $R

    ./a2 input.txt knn        > /dev/null 2> _t_knn.txt
    if [ $? -ne 0 ]; then
        echo "  ERROR running knn"
        echo "$group,$label,$n,$k,$T,$dist,$R,ERROR,ERROR,ERROR,ERROR,N/A,N/A" >> $RESULTS
        return
    fi
    ./a2 input.txt approx_knn > /dev/null 2> _t_approx.txt
    ./a2 input.txt kmeans     > /dev/null 2> _t_kmeans.txt

    cat _t_knn.txt _t_approx.txt _t_kmeans.txt

    knn_s=$(grep    "^KNN:"        _t_knn.txt    | awk '{print $2}')
    approx_s=$(grep "^Approx KNN:" _t_approx.txt | awk '{print $3}')
    kmeans_s=$(grep "^K-Means:"    _t_kmeans.txt | awk '{print $2}')

    python3 - << PYEOF
def mae(f1, f2):
    try:
        a = [int(l.split()[3]) for l in open(f1)]
        b = [int(l.split()[3]) for l in open(f2)]
        return sum(abs(x-y) for x,y in zip(a,b)) / len(a)
    except:
        return None

approx_mae     = mae('approx_knn.txt', 'knn.txt')
knn_seq_mae    = None
kmeans_seq_mae = None

if $run_seq:
    import subprocess, sys as _sys
    subprocess.run([_sys.executable, 'sequential.py', 'input.txt'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    knn_seq_mae    = mae('knn.txt',    'knn_seq.txt')
    kmeans_seq_mae = mae('kmeans.txt', 'kmeans_seq.txt')
    print(f"  [seq] knn_mae={knn_seq_mae:.4f}  kmeans_mae={kmeans_seq_mae:.4f}")

def fmt(v): return f'{v:.4f}' if v is not None else 'N/A'
print(f"  approx_mae_vs_knn={fmt(approx_mae)}")

with open('$RESULTS', 'a') as f:
    f.write(f'$group,$label,$n,$k,$T,$dist,$R,$knn_s,$approx_s,$kmeans_s,'
            f'{fmt(approx_mae)},{fmt(knn_seq_mae)},{fmt(kmeans_seq_mae)}\n')
PYEOF
    echo ""
}

# ===========================================================================
#  GROUP A — Scalability with N  (k=32, T=20, uniform)
# ===========================================================================
echo "========================================"
echo " GROUP A: Vary N  (k=32, T=20, uniform)"
echo "========================================"
for n in 1000 5000 10000 25000 50000 100000; do
    run_case "vary_n" "n${n}" $n 32 20 uniform 0 10000
done

# ===========================================================================
#  GROUP B — Scalability with K  (n=10000, T=20, uniform)
# ===========================================================================
echo "========================================"
echo " GROUP B: Vary K  (n=10000, T=20, uniform)"
echo "========================================"
for k in 1 4 8 16 32 64 128; do
    run_case "vary_k" "k${k}" 10000 $k 20 uniform 0 10000
done

# ===========================================================================
#  GROUP C — K-Means convergence  (n=10000, k=32, uniform)
# ===========================================================================
echo "========================================"
echo " GROUP C: Vary T  (n=10000, k=32, uniform)"
echo "========================================"
for T in 1 5 10 20 50; do
    run_case "vary_T" "T${T}" 10000 32 $T uniform 0 10000
done

# ===========================================================================
#  GROUP D — Data distributions  (n=10000, k=32, T=20)
# ===========================================================================
echo "========================================"
echo " GROUP D: Distributions  (n=10000, k=32, T=20)"
echo "========================================"
for dist in uniform dense clustered skewed; do
    run_case "dist" "$dist" 10000 32 20 $dist 0 10000
done

# ===========================================================================
#  GROUP E — Correctness vs sequential  (n=1000, seq is feasible here)
# ===========================================================================
echo "========================================"
echo " GROUP E: Correctness  (n=1000, vs sequential)"
echo "========================================"
for k in 1 10 32 64 128; do
    run_case "correctness" "k${k}_uniform" 1000 $k 20 uniform 1 10000
done
for dist in dense clustered skewed; do
    run_case "correctness" "k32_${dist}" 1000 32 20 $dist 1 10000
done

# ===========================================================================
#  GROUP F — Large scale stress tests
# ===========================================================================
echo "========================================"
echo " GROUP F: Large scale"
echo "========================================"
run_case "large" "n100k_k128_T50"      100000 128 50 uniform 0 10000
run_case "large" "n100k_k32_uniform"   100000  32 20 uniform 0 10000
run_case "large" "n100k_k32_dense"     100000  32 20 dense   0 10000
run_case "large" "n100k_k32_clustered" 100000  32 20 clustered 0 10000

# ===========================================================================
#  GROUP G — Coordinate range variation  (n=10000, k=32, T=20, uniform)
# ===========================================================================
echo "========================================"
echo " GROUP G: Vary coordinate range  (n=10000, k=32, T=20, uniform)"
echo "========================================"
for R in 100 1000 10000 100000 1000000; do
    run_case "vary_range" "R${R}" 10000 32 20 uniform 0 $R
done

# ---------------------------------------------------------------------------
#  Summary table
# ---------------------------------------------------------------------------
echo "========================================"
echo " RESULTS SUMMARY"
echo "========================================"
python3 - << 'PYEOF'
import csv, sys

try:
    rows = list(csv.DictReader(open('results.csv')))
except:
    print("results.csv not found"); sys.exit()

groups = {}
for r in rows:
    groups.setdefault(r['group'], []).append(r)

for g, rs in groups.items():
    print(f"\n{'='*68}")
    print(f"  {g}")
    print(f"{'='*68}")
    print(f"  {'label':<28} {'range':>9} {'knn_s':>9} {'approx_s':>10} {'kmeans_s':>10} {'approx_mae':>11}")
    print(f"  {'-'*28} {'-'*9} {'-'*9} {'-'*10} {'-'*10} {'-'*11}")
    for r in rs:
        print(f"  {r['label']:<28} {r['range']:>9} {r['knn_s']:>9} {r['approx_s']:>10} {r['kmeans_s']:>10} {r['approx_mae_vs_knn']:>11}")
PYEOF

echo ""
echo "Full data: results.csv"
echo "========================================"
echo " DONE"
echo "========================================"
