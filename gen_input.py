import sys
import random

def generate_dataset(n, k, T, output_file="input.txt", seed=None):
    if seed is not None:
        random.seed(seed)
    assert 1 <= n <= 100000
    assert 1 <= k <= 128
    assert 1 <= T <= 50
    used_points = set()
    with open(output_file, "w") as f:
        f.write(f"{n}\n")
        f.write(f"{k}\n")
        f.write(f"{T}\n")
        while len(used_points) < n:
            x = random.randint(-10000, 10000)
            y = random.randint(-10000, 10000)
            z = random.randint(-10000, 10000)
            if (x, y, z) in used_points:
                continue
            used_points.add((x, y, z))
            I = random.randint(0, 255)
            f.write(f"{x} {y} {z} {I}\n")
    print(f"Dataset written to {output_file}")

if __name__ == "__main__":
    n = int(sys.argv[1])
    k = int(sys.argv[2])
    T = int(sys.argv[3])
    out = sys.argv[4] if len(sys.argv) > 4 else "input.txt"
    generate_dataset(n, k, T, out)