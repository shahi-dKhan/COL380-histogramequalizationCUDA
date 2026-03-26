import random
import argparse

# ---------------------------------------------------------------------------
#  Distributions  (all accept a coordinate range R; coords stay in [-R, R])
# ---------------------------------------------------------------------------
def _gen_pts(n, sample_fn, seed):
    if seed is not None:
        random.seed(seed)
    used = set()
    pts = []
    while len(pts) < n:
        x, y, z = sample_fn()
        if (x, y, z) in used:
            continue
        used.add((x, y, z))
        pts.append((x, y, z, random.randint(0, 255)))
    return pts

def generate_uniform(n, seed=None, R=10000):
    """Uniform random in [-R, R]^3."""
    return _gen_pts(n, lambda: (
        random.randint(-R, R),
        random.randint(-R, R),
        random.randint(-R, R)), seed)

def generate_dense(n, seed=None, R=10000):
    """Concentrated in [-R//10, R//10]^3 — many close neighbours."""
    r = max(1, R // 10)
    return _gen_pts(n, lambda: (
        random.randint(-r, r),
        random.randint(-r, r),
        random.randint(-r, r)), seed)

def generate_clustered(n, seed=None, R=10000):
    """8 tight clusters at the corners of a cube — ideal for K-Means."""
    if seed is not None:
        random.seed(seed)
    centre_dist = int(R * 0.70)          # cluster centres at ±70 % of range
    cluster_r   = max(1, R // 8)         # radius of each cluster
    centres = [(sx * centre_dist, sy * centre_dist, sz * centre_dist)
               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    used = set()
    pts = []
    while len(pts) < n:
        cx, cy, cz = random.choice(centres)
        x = max(-R, min(R, cx + random.randint(-cluster_r, cluster_r)))
        y = max(-R, min(R, cy + random.randint(-cluster_r, cluster_r)))
        z = max(-R, min(R, cz + random.randint(-cluster_r, cluster_r)))
        if (x, y, z) in used:
            continue
        used.add((x, y, z))
        pts.append((x, y, z, random.randint(0, 255)))
    return pts

def generate_skewed(n, seed=None, R=10000):
    """Half points dense in [-R//20, R//20]^3, half uniform in [-R, R]^3."""
    if seed is not None:
        random.seed(seed)
    r = max(1, R // 20)
    used = set()
    pts = []
    dense_target = n // 2
    dense_done   = 0
    while len(pts) < n:
        if dense_done < dense_target:
            x = random.randint(-r, r)
            y = random.randint(-r, r)
            z = random.randint(-r, r)
        else:
            x = random.randint(-R, R)
            y = random.randint(-R, R)
            z = random.randint(-R, R)
        if (x, y, z) in used:
            continue
        used.add((x, y, z))
        pts.append((x, y, z, random.randint(0, 255)))
        if dense_done < dense_target:
            dense_done += 1
    return pts

_GENERATORS = {
    'uniform':   generate_uniform,
    'dense':     generate_dense,
    'clustered': generate_clustered,
    'skewed':    generate_skewed,
}

# ---------------------------------------------------------------------------
#  Writer
# ---------------------------------------------------------------------------
def generate_dataset(n, k, T, output_file="input.txt", seed=None, dist="uniform", R=10000):
    assert 1 <= n <= 100000, f"n={n} out of range"
    assert 1 <= k <= 128,    f"k={k} out of range"
    assert 1 <= T <= 50,     f"T={T} out of range"
    assert dist in _GENERATORS, f"Unknown dist '{dist}'"

    pts = _GENERATORS[dist](n, seed, R=R)

    with open(output_file, "w") as f:
        f.write(f"{n}\n{k}\n{T}\n")
        for x, y, z, I in pts:
            f.write(f"{x} {y} {z} {I}\n")
    print(f"Dataset written to {output_file}  [n={n}, k={k}, T={T}, dist={dist}, seed={seed}, R={R}]")

# ---------------------------------------------------------------------------
#  CLI  (backward-compatible: gen_input.py N K T [outfile] [--dist X] [--seed N])
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate 3-D point-cloud test input")
    p.add_argument("n",      type=int)
    p.add_argument("k",      type=int)
    p.add_argument("T",      type=int)
    p.add_argument("output", nargs="?", default="input.txt")
    p.add_argument("--dist", default="uniform",
                   choices=list(_GENERATORS.keys()),
                   help="Point distribution (default: uniform)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility")
    p.add_argument("--range", type=int, default=10000,
                   help="Max coordinate value; points in [-range, range]^3 (default: 10000)")
    args = p.parse_args()
    generate_dataset(args.n, args.k, args.T, args.output,
                     seed=args.seed, dist=args.dist, R=args.range)
