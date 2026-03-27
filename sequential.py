"""
Sequential reference implementations for all three approaches.
Used to verify CUDA output correctness.
Run as: python3 sequential.py input.txt
Outputs: knn_seq.txt, approx_knn_seq.txt, kmeans_seq.txt
"""

import sys
import math
import random

# ---------------------------------------------------------------------------
#  Read input
# ---------------------------------------------------------------------------
def read_input(path):
    with open(path) as f:
        n = int(f.readline())
        k = int(f.readline())
        T = int(f.readline())
        pts = [list(map(int, f.readline().split())) for _ in range(n)]
    return n, k, T, pts

# ---------------------------------------------------------------------------
#  Equalization helper
# ---------------------------------------------------------------------------
def equalize(hist, orig, m):
    cdf, cdf_min = 0, -1
    for v in range(256):
        cdf += hist[v]
        if cdf > 0 and cdf_min < 0:
            cdf_min = cdf
        if v == orig:
            if m == cdf_min:
                return orig
            val = int(math.floor((cdf - cdf_min) / (m - cdf_min) * 255.0))
            return max(0, min(255, val))
    return orig

# ---------------------------------------------------------------------------
#  1. Exact KNN
# ---------------------------------------------------------------------------
def run_knn(n, k, pts):
    result = []
    for i in range(n):
        if k == 0:
            result.append(pts[i][3])
            continue

        # find k nearest neighbours excluding self
        dists = []
        for j in range(n):
            if j == i: continue
            dx = pts[j][0] - pts[i][0]
            dy = pts[j][1] - pts[i][1]
            dz = pts[j][2] - pts[i][2]
            dists.append((dx*dx + dy*dy + dz*dz, j))
        dists.sort()
        neighbours = [pts[j][3] for _, j in dists[:k]]

        # histogram over self + k neighbours
        neighbourhood = [pts[i][3]] + neighbours
        hist = [0] * 256
        for v in neighbourhood:
            hist[v] += 1

        m = k + 1
        result.append(equalize(hist, pts[i][3], m))

    return result

# ---------------------------------------------------------------------------
#  2. Approximate KNN  (same as exact for reference — we compare CUDA
#     approx against CUDA exact, not sequential approx)
#     So we just output exact KNN here as the reference baseline.
# ---------------------------------------------------------------------------
def run_approx_knn(n, k, pts):
    return run_knn(n, k, pts)

# ---------------------------------------------------------------------------
#  3. K-Means
# ---------------------------------------------------------------------------
def dist2(a, b):
    dx, dy, dz = a[0]-b[0], a[1]-b[1], a[2]-b[2]
    return dx*dx + dy*dy + dz*dz

def run_kmeans(n, k, T, pts):
    if k == 0:
        return [p[3] for p in pts]

    # Initial centroids = first k points (integer coords, matching CUDA)
    centroids = [[pts[c][0], pts[c][1], pts[c][2]] for c in range(k)]

    assign = [-1] * n

    for _ in range(T):
        # Assignment — tie-break lexicographically by centroid coords,
        # matching the CUDA kmeans_assign_kernel tie-breaking rule.
        new_assign = []
        for i in range(n):
            best, bi = float('inf'), 0
            for c in range(k):
                dx = centroids[c][0] - pts[i][0]
                dy = centroids[c][1] - pts[i][1]
                dz = centroids[c][2] - pts[i][2]
                d2 = dx*dx + dy*dy + dz*dz
                if d2 < best or (d2 == best and (
                        centroids[c][0] < centroids[bi][0] or
                        (centroids[c][0] == centroids[bi][0] and centroids[c][1] < centroids[bi][1]) or
                        (centroids[c][0] == centroids[bi][0] and centroids[c][1] == centroids[bi][1]
                         and centroids[c][2] < centroids[bi][2]))):
                    best, bi = d2, c
            new_assign.append(bi)

        # Convergence check
        if new_assign == assign:
            break
        assign = new_assign

        # Update centroids — integer division matching CUDA kmeans_update_kernel
        sums = [[0, 0, 0] for _ in range(k)]
        counts = [0] * k
        for i in range(n):
            c = assign[i]
            sums[c][0] += pts[i][0]
            sums[c][1] += pts[i][1]
            sums[c][2] += pts[i][2]
            counts[c] += 1
        for c in range(k):
            if counts[c] > 0:
                # Floor division — matches CUDA's unsigned-shift accumulation:
                # CUDA does floor(sum/cnt) via unsigned offset trick, not truncation.
                centroids[c] = [sums[c][d] // counts[c] for d in range(3)]

    # Build per-cluster histograms
    hists   = [[0] * 256 for _ in range(k)]
    sizes   = [0] * k
    for i in range(n):
        c = assign[i]
        hists[c][pts[i][3]] += 1
        sizes[c] += 1

    result = []
    for i in range(n):
        c = assign[i]
        result.append(equalize(hists[c], pts[i][3], sizes[c]))

    return result

# ---------------------------------------------------------------------------
#  Write output
# ---------------------------------------------------------------------------
def write_output(path, pts, result):
    with open(path, 'w') as f:
        for i, p in enumerate(pts):
            f.write(f"{p[0]} {p[1]} {p[2]} {result[i]}\n")

# ---------------------------------------------------------------------------
#  MAE
# ---------------------------------------------------------------------------
def mae(file1, file2):
    a = [int(l.split()[3]) for l in open(file1)]
    b = [int(l.split()[3]) for l in open(file2)]
    return sum(abs(x-y) for x,y in zip(a,b)) / len(a)

# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    input_file = sys.argv[1]
    n, k, T, pts = read_input(input_file)

    print(f"  Running sequential KNN    (n={n}, k={k})...")
    knn_result = run_knn(n, k, pts)
    write_output("knn_seq.txt", pts, knn_result)

    print(f"  Running sequential K-Means (n={n}, k={k}, T={T})...")
    km_result = run_kmeans(n, k, T, pts)
    write_output("kmeans_seq.txt", pts, km_result)

    # Compare CUDA outputs vs sequential
    print(f"  [knn]    MAE vs sequential = {mae('knn.txt',    'knn_seq.txt'):.4f}")
    print(f"  [approx] MAE vs sequential = {mae('approx_knn.txt', 'knn_seq.txt'):.4f}")
    print(f"  [kmeans] MAE vs sequential = {mae('kmeans.txt', 'kmeans_seq.txt'):.4f}")