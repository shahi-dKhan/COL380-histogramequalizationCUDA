#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <cfloat>
#include <climits>

#define MY_LLONG_MAX 9223372036854775807LL
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if(err != cudaSuccess){ \
            throw std::runtime_error("CUDA Error at " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " - " + cudaGetErrorString(err)); \
        } \
    } while(0)

template<typename T>
struct DeviceBuffer{
    T* ptr = nullptr;
    int size = 0;
    explicit DeviceBuffer(int size): size(size){
        CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(T)));
    }
    ~DeviceBuffer(){ if(ptr) cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr), size(other.size){
        other.ptr = nullptr;
    }
    void upload(const std::vector<int>& hostData){
        CUDA_CHECK(cudaMemcpy(ptr, hostData.data(), size * sizeof(T), cudaMemcpyHostToDevice));
    }
    void download(std::vector<int>& hostData){
        CUDA_CHECK(cudaMemcpy(hostData.data(), ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
    }
    void zero(){
        CUDA_CHECK(cudaMemset(ptr, 0, size * sizeof(T)));
    }
    void copy_from(const DeviceBuffer& other){
        if(size != other.size) throw std::runtime_error("Size mismatch in copy_from");
        CUDA_CHECK(cudaMemcpy(ptr, other.ptr, size * sizeof(T), cudaMemcpyDeviceToDevice));
    }
    T download_one() const {
        T val;
        CUDA_CHECK(cudaMemcpy(&val, ptr, sizeof(T), cudaMemcpyDeviceToHost));
        return val;
    }
};

constexpr int INTENSITY_LEVELS = 256;
constexpr int BLOCK_SIZE       = 128;
constexpr int MAX_K            = 128;

// ---------------------------------------------------------------------------
//  Toggle: 1 = shared-memory tiled KNN (128x bandwidth reduction for the
//              all-pairs distance loop), 0 = original non-tiled version.
//  Can be overridden at compile time: nvcc -DUSE_TILED_KNN=0 ...
// ---------------------------------------------------------------------------
#ifndef USE_TILED_KNN
#define USE_TILED_KNN 1
#endif

// ---------------------------------------------------------------------------
//  Toggle: 1 = spatial-hash approximate KNN.  Hash table of size 2N with
//              chaining replaces the flat 3D grid array, eliminating the 10M
//              cell cap and keeping the table (ht_head + ht_next) at ~2.4 MB
//              for N=100K regardless of coordinate range.
//          0 = original uniform-grid version.
//  Can be overridden at compile time: nvcc -DUSE_SPATIAL_HASH_APPROX=1 ...
// ---------------------------------------------------------------------------
#ifndef USE_SPATIAL_HASH_APPROX
#define USE_SPATIAL_HASH_APPROX 0
#endif

struct PointCloud {
    int n, k, T;
    std::vector<int> x, y, z, intensity;
};

PointCloud read_input(const std::string &path)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    PointCloud pc;
    f >> pc.n >> pc.k >> pc.T;
    pc.x.resize(pc.n); pc.y.resize(pc.n);
    pc.z.resize(pc.n); pc.intensity.resize(pc.n);
    for (int i = 0; i < pc.n; ++i)
        f >> pc.x[i] >> pc.y[i] >> pc.z[i] >> pc.intensity[i];
    return pc;
}

void write_output(const std::string      &path,
                  const PointCloud       &pc,
                  const std::vector<int> &new_intensity)
{
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    for (int i = 0; i < pc.n; ++i)
        f << pc.x[i] << ' ' << pc.y[i] << ' '
          << pc.z[i] << ' ' << new_intensity[i] << '\n';
}

__device__ __forceinline__ int equalize_point(const int * __restrict__ hist, int orig, int m)
{
    int cdf = 0, cdf_min = -1;
    for (int v = 0; v < INTENSITY_LEVELS; ++v) {
        cdf += hist[v];
        if (cdf > 0 && cdf_min < 0) cdf_min = cdf;
        if (v == orig) {
            if (m == cdf_min) return orig;
            int val = __float2int_rn(
                static_cast<float>(cdf - cdf_min) /
                static_cast<float>(m   - cdf_min) * (INTENSITY_LEVELS - 1));
            return max(0, min(INTENSITY_LEVELS - 1, val));
        }
    }
    return orig;
}

__device__ __forceinline__ void heap_bubble_up(long long *dist, int *idx, int pos)
{
    while (pos > 0) {
        int par = (pos - 1) >> 1;
        if (dist[par] < dist[pos]) {
            long long td = dist[par]; dist[par] = dist[pos]; dist[pos] = td;
            int       ti = idx [par]; idx [par] = idx [pos]; idx [pos] = ti;
            pos = par;
        } else break;
    }
}

__device__ __forceinline__ void heap_sift_down(long long *dist, int *idx, int k)
{
    int pos = 0;
    while (true) {
        int l = 2*pos+1, r = 2*pos+2, largest = pos;
        if (l < k && dist[l] > dist[largest]) largest = l;
        if (r < k && dist[r] > dist[largest]) largest = r;
        if (largest == pos) break;
        long long td = dist[pos]; dist[pos] = dist[largest]; dist[largest] = td;
        int       ti = idx [pos]; idx [pos] = idx [largest]; idx [largest] = ti;
        pos = largest;
    }
}

#define HEAP_INSERT(dist, idx, heap_size, k, d2, j)  \
    if ((heap_size) < (k)) {                          \
        (dist)[(heap_size)] = (d2);                   \
        (idx)[(heap_size)] = (j);                     \
        heap_bubble_up((dist), (idx), (heap_size));   \
        (heap_size)++;                                \
    } else if ((d2) < (dist)[0]) {                    \
        (dist)[0] = (d2); (idx)[0] = (j);             \
        heap_sift_down((dist), (idx), (k));            \
    }

// ===========================================================================
//  1. EXACT KNN
//  Each thread scans all N points (skipping self), maintains a max-heap of
//  k nearest in registers, builds histogram over self + k neighbours,
//  equalizes with m = k+1.
// ===========================================================================
#if USE_TILED_KNN
// Tiled version: block cooperatively loads BLOCK_SIZE points into shared
// memory per tile, giving each thread BLOCK_SIZE reuses per load instead of 1.
// Shared memory cost: 3 * BLOCK_SIZE * 4 = 1536 bytes — negligible.
// All threads (including out-of-bounds ones) must participate in __syncthreads.
__global__ void knn_kernel(const int * __restrict__ xs,
                           const int * __restrict__ ys,
                           const int * __restrict__ zs,
                           const int * __restrict__ intensity,
                           int       * __restrict__ out,
                           int n, int k)
{
    __shared__ int s_xs[BLOCK_SIZE], s_ys[BLOCK_SIZE], s_zs[BLOCK_SIZE];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    // k==0: uniform branch — all threads exit together, no syncthreads hazard.
    if (k == 0) {
        if (i < n) out[i] = intensity[i];
        return;
    }

    const bool active = (i < n);
    const int qx = active ? xs[i] : 0;
    const int qy = active ? ys[i] : 0;
    const int qz = active ? zs[i] : 0;

    long long heap_dist[MAX_K];
    int       heap_idx [MAX_K];
    int       heap_size = 0;

    for (int tile_start = 0; tile_start < n; tile_start += BLOCK_SIZE) {
        // Cooperative load: every thread loads one element of the tile.
        int j = tile_start + threadIdx.x;
        s_xs[threadIdx.x] = (j < n) ? xs[j] : 0;
        s_ys[threadIdx.x] = (j < n) ? ys[j] : 0;
        s_zs[threadIdx.x] = (j < n) ? zs[j] : 0;
        __syncthreads();

        if (active) {
            const int tile_len = min(BLOCK_SIZE, n - tile_start);
            for (int t = 0; t < tile_len; ++t) {
                const int jj = tile_start + t;
                if (jj == i) continue;
                long long dx = s_xs[t]-qx, dy = s_ys[t]-qy, dz = s_zs[t]-qz;
                long long d2 = dx*dx + dy*dy + dz*dz;
                HEAP_INSERT(heap_dist, heap_idx, heap_size, k, d2, jj);
            }
        }
        __syncthreads();
    }

    if (!active) return;

    int hist[INTENSITY_LEVELS] = {};
    hist[intensity[i]]++;
    for (int t = 0; t < heap_size; ++t) hist[intensity[heap_idx[t]]]++;
    out[i] = equalize_point(hist, intensity[i], heap_size + 1);
}

#else  // USE_TILED_KNN == 0: original non-tiled version

__global__ void knn_kernel(const int * __restrict__ xs,
                           const int * __restrict__ ys,
                           const int * __restrict__ zs,
                           const int * __restrict__ intensity,
                           int       * __restrict__ out,
                           int n, int k)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (k == 0) { out[i] = intensity[i]; return; }

    const int qx = xs[i], qy = ys[i], qz = zs[i];

    long long heap_dist[MAX_K];
    int       heap_idx [MAX_K];
    int       heap_size = 0;

    for (int j = 0; j < n; ++j) {
        if (j == i) continue;
        long long dx = xs[j]-qx, dy = ys[j]-qy, dz = zs[j]-qz;
        long long d2 = dx*dx + dy*dy + dz*dz;
        HEAP_INSERT(heap_dist, heap_idx, heap_size, k, d2, j);
    }

    int hist[INTENSITY_LEVELS] = {};
    hist[intensity[i]]++;
    for (int t = 0; t < heap_size; ++t) hist[intensity[heap_idx[t]]]++;
    out[i] = equalize_point(hist, intensity[i], heap_size + 1);
}
#endif  // USE_TILED_KNN

std::vector<int> run_knn(const PointCloud &pc)
{
    DeviceBuffer<int> d_x(pc.n), d_y(pc.n), d_z(pc.n), d_I(pc.n), d_out(pc.n);
    d_x.upload(pc.x); d_y.upload(pc.y); d_z.upload(pc.z); d_I.upload(pc.intensity);

    knn_kernel<<<(pc.n + BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        d_x.ptr, d_y.ptr, d_z.ptr, d_I.ptr, d_out.ptr, pc.n, pc.k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> result(pc.n);
    d_out.download(result);
    return result;
}

// ===========================================================================
//  2. APPROXIMATE KNN -- Uniform Grid / Voxel Hashing
//
//  Build (host, OpenMP):
//    Choose cell size r so ~k/27 points fall per cell on average.
//    Counting-sort points into cells.
//
//  Query (GPU):
//    Dynamic expanding window: start at radius=1 (3x3x3), expand shell by
//    shell until heap has k neighbours or max_radius is reached.
//    Warp-level early exit via __all_sync when all threads in warp are full.
// ===========================================================================
__global__ void approx_knn_kernel(const int * __restrict__ xs,
                                   const int * __restrict__ ys,
                                   const int * __restrict__ zs,
                                   const int * __restrict__ intensity,
                                   const int * __restrict__ sorted_ids,
                                   const int * __restrict__ cell_start,
                                   const int * __restrict__ cell_end,
                                   int       * __restrict__ out,
                                   int min_x, int min_y, int min_z,
                                   float inv_r,
                                   int nx, int ny, int nz,
                                   int n, int k, int max_radius)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (k == 0) { out[i] = intensity[i]; return; }

    const int qx = xs[i], qy = ys[i], qz = zs[i];
    const int cx = min(nx-1, static_cast<int>((qx - min_x) * inv_r));
    const int cy = min(ny-1, static_cast<int>((qy - min_y) * inv_r));
    const int cz = min(nz-1, static_cast<int>((qz - min_z) * inv_r));

    long long heap_dist[MAX_K];
    int       heap_idx [MAX_K];
    int       heap_size = 0;

    int prev = -1;
    for (int radius = 1; radius <= max_radius; ++radius) {
        for (int dz = -radius; dz <= radius; ++dz) {
            int nz_ = cz + dz; if (nz_ < 0 || nz_ >= nz) continue;
            for (int dy = -radius; dy <= radius; ++dy) {
                int ny_ = cy + dy; if (ny_ < 0 || ny_ >= ny) continue;
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (dx >= -prev && dx <= prev &&
                        dy >= -prev && dy <= prev &&
                        dz >= -prev && dz <= prev) continue;
                    int nx_ = cx + dx; if (nx_ < 0 || nx_ >= nx) continue;
                    const int cell = nz_ * ny * nx + ny_ * nx + nx_;
                    for (int s = cell_start[cell]; s < cell_end[cell]; ++s) {
                        const int j = sorted_ids[s];
                        if (j == i) continue;
                        long long ddx = xs[j]-qx, ddy = ys[j]-qy, ddz = zs[j]-qz;
                        long long d2  = ddx*ddx + ddy*ddy + ddz*ddz;
                        HEAP_INSERT(heap_dist, heap_idx, heap_size, k, d2, j);
                    }
                }
            }
        }
        prev = radius;
        if (__all_sync(__activemask(), heap_size >= k)) break;
    }

    int hist[INTENSITY_LEVELS] = {};
    hist[intensity[i]]++;
    for (int t = 0; t < heap_size; ++t) hist[intensity[heap_idx[t]]]++;
    out[i] = equalize_point(hist, intensity[i], heap_size + 1);
}

// ---------------------------------------------------------------------------
//  Spatial hash for 3D integer cell coordinates.
//  Avalanche mixing after XOR-combine gives good distribution over power-of-2
//  table sizes; caller applies result & ht_mask.
// ---------------------------------------------------------------------------
__device__ __host__ __forceinline__
unsigned int spatial_hash(int cx, int cy, int cz)
{
    unsigned int h = (unsigned int)cx * 73856093u
                   ^ (unsigned int)cy * 19349663u
                   ^ (unsigned int)cz * 83492791u;
    h ^= h >> 16;
    h *= 0x45d9f3bu;
    h ^= h >> 16;
    return h;
}

// ---------------------------------------------------------------------------
//  APPROX KNN — Spatial Hash kernel
//
//  ht_head[ht_size] + ht_next[n] form a chained hash table keyed on cell
//  coordinates.  For each shell cell (qcx+dx, qcy+dy, qcz+dz):
//    - Hash the cell coords → bucket index.
//    - Walk the chain; skip entries with different cell coords (hash collision)
//      and self.
//    - No bounds check needed: out-of-range cells have no entries (head = -1).
// ---------------------------------------------------------------------------
__global__ void approx_knn_hash_kernel(
    const int * __restrict__ xs,
    const int * __restrict__ ys,
    const int * __restrict__ zs,
    const int * __restrict__ intensity,
    const int * __restrict__ pt_cx,
    const int * __restrict__ pt_cy,
    const int * __restrict__ pt_cz,
    const int * __restrict__ ht_head,
    const int * __restrict__ ht_next,
    int       * __restrict__ out,
    int ht_mask, int n, int k, int max_radius)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (k == 0) { out[i] = intensity[i]; return; }

    const int qx = xs[i], qy = ys[i], qz = zs[i];
    const int qcx = pt_cx[i], qcy = pt_cy[i], qcz = pt_cz[i];

    long long heap_dist[MAX_K];
    int       heap_idx [MAX_K];
    int       heap_size = 0;

    int prev = -1;
    for (int radius = 1; radius <= max_radius; ++radius) {
        for (int dz = -radius; dz <= radius; ++dz) {
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    // Only process the new shell layer (skip inner cube).
                    if (dx >= -prev && dx <= prev &&
                        dy >= -prev && dy <= prev &&
                        dz >= -prev && dz <= prev) continue;

                    const int tcx = qcx + dx;
                    const int tcy = qcy + dy;
                    const int tcz = qcz + dz;
                    const int h = (int)(spatial_hash(tcx, tcy, tcz) & (unsigned int)ht_mask);

                    for (int j = ht_head[h]; j != -1; j = ht_next[j]) {
                        // Resolve hash collision: verify cell coords match.
                        if (pt_cx[j] != tcx || pt_cy[j] != tcy || pt_cz[j] != tcz) continue;
                        if (j == i) continue;
                        long long ddx = xs[j]-qx, ddy = ys[j]-qy, ddz = zs[j]-qz;
                        HEAP_INSERT(heap_dist, heap_idx, heap_size, k,
                                    ddx*ddx + ddy*ddy + ddz*ddz, j);
                    }
                }
            }
        }
        prev = radius;
        if (__all_sync(__activemask(), heap_size >= k)) break;
    }

    int hist[INTENSITY_LEVELS] = {};
    hist[intensity[i]]++;
    for (int t = 0; t < heap_size; ++t) hist[intensity[heap_idx[t]]]++;
    out[i] = equalize_point(hist, intensity[i], heap_size + 1);
}

// ---------------------------------------------------------------------------
//  Host-side driver: uniform grid (original)
// ---------------------------------------------------------------------------
static std::vector<int> run_approx_knn_grid(const PointCloud &pc)
{
    const int n = pc.n, k = pc.k;

    int min_x = INT_MAX, min_y = INT_MAX, min_z = INT_MAX;
    int max_x = INT_MIN, max_y = INT_MIN, max_z = INT_MIN;
    #pragma omp parallel for schedule(static) \
        reduction(min: min_x, min_y, min_z)   \
        reduction(max: max_x, max_y, max_z)
    for (int i = 0; i < n; ++i) {
        min_x = std::min(min_x, pc.x[i]); max_x = std::max(max_x, pc.x[i]);
        min_y = std::min(min_y, pc.y[i]); max_y = std::max(max_y, pc.y[i]);
        min_z = std::min(min_z, pc.z[i]); max_z = std::max(max_z, pc.z[i]);
    }

    const long long vol = static_cast<long long>(max_x - min_x + 1)
                        * static_cast<long long>(max_y - min_y + 1)
                        * static_cast<long long>(max_z - min_z + 1);
    const float pts_per_cell = std::max(1.0f, static_cast<float>(k) / 27.0f);
    float r = std::cbrt(static_cast<float>(vol) / (static_cast<float>(n) / pts_per_cell));
    r = std::max(r, 1.0f);
    const float inv_r = 1.0f / r;

    int nx = std::max(1, static_cast<int>((max_x - min_x) * inv_r) + 1);
    int ny = std::max(1, static_cast<int>((max_y - min_y) * inv_r) + 1);
    int nz = std::max(1, static_cast<int>((max_z - min_z) * inv_r) + 1);
    if (static_cast<long long>(nx)*ny*nz > 10000000LL) {
        float scale = std::cbrt(10000000.0f / static_cast<float>(nx*ny*nz));
        nx = std::max(1, static_cast<int>(nx * scale));
        ny = std::max(1, static_cast<int>(ny * scale));
        nz = std::max(1, static_cast<int>(nz * scale));
    }
    const int total_cells = nx * ny * nz;

    std::vector<int> cell_id(n), count(total_cells, 0);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        int cx = std::min(static_cast<int>((pc.x[i] - min_x) * inv_r), nx-1);
        int cy = std::min(static_cast<int>((pc.y[i] - min_y) * inv_r), ny-1);
        int cz = std::min(static_cast<int>((pc.z[i] - min_z) * inv_r), nz-1);
        cell_id[i] = cz * ny * nx + cy * nx + cx;
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        #pragma omp atomic
        ++count[cell_id[i]];
    }

    std::vector<int> cell_start(total_cells), cell_end(total_cells);
    cell_start[0] = 0;
    for (int c = 1; c < total_cells; ++c)
        cell_start[c] = cell_start[c-1] + count[c-1];
    for (int c = 0; c < total_cells; ++c)
        cell_end[c] = cell_start[c] + count[c];

    std::vector<int> sorted_ids(n), tmp(total_cells, 0);
    for (int i = 0; i < n; ++i)
        sorted_ids[cell_start[cell_id[i]] + tmp[cell_id[i]]++] = i;

    float cells_needed = static_cast<float>(k) * total_cells / static_cast<float>(n);
    int max_radius = static_cast<int>(std::cbrt(cells_needed) / 2.0f) + 3;
    max_radius = std::min(max_radius, std::max({nx, ny, nz}));
    max_radius = std::max(max_radius, 1);

    DeviceBuffer<int> d_x(n), d_y(n), d_z(n), d_I(n), d_out(n);
    DeviceBuffer<int> d_sorted(n), d_cs(total_cells), d_ce(total_cells);
    d_x.upload(pc.x); d_y.upload(pc.y); d_z.upload(pc.z); d_I.upload(pc.intensity);
    d_sorted.upload(sorted_ids);
    d_cs.upload(cell_start); d_ce.upload(cell_end);

    approx_knn_kernel<<<(n + BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        d_x.ptr, d_y.ptr, d_z.ptr, d_I.ptr,
        d_sorted.ptr, d_cs.ptr, d_ce.ptr, d_out.ptr,
        min_x, min_y, min_z, inv_r, nx, ny, nz, n, k, max_radius);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> result(n);
    d_out.download(result);
    return result;
}

// ---------------------------------------------------------------------------
//  Host-side driver: spatial hash
// ---------------------------------------------------------------------------
static std::vector<int> run_approx_knn_hash(const PointCloud &pc)
{
    const int n = pc.n, k = pc.k;

    // Bounding box (used only for cell-size selection and cell coord origin).
    int min_x = *std::min_element(pc.x.begin(), pc.x.end());
    int max_x = *std::max_element(pc.x.begin(), pc.x.end());
    int min_y = *std::min_element(pc.y.begin(), pc.y.end());
    int max_y = *std::max_element(pc.y.begin(), pc.y.end());
    int min_z = *std::min_element(pc.z.begin(), pc.z.end());
    int max_z = *std::max_element(pc.z.begin(), pc.z.end());

    // Same cell-size heuristic as the grid version: target k/27 pts/cell.
    const long long vol = (long long)(max_x - min_x + 1)
                        * (long long)(max_y - min_y + 1)
                        * (long long)(max_z - min_z + 1);
    const float pts_per_cell = std::max(1.0f, static_cast<float>(k) / 27.0f);
    float r = std::cbrt(static_cast<float>(vol) / (static_cast<float>(n) / pts_per_cell));
    r = std::max(r, 1.0f);
    const float inv_r = 1.0f / r;

    // Cell coordinates for every point (non-negative since origin = min).
    std::vector<int> pt_cx(n), pt_cy(n), pt_cz(n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        pt_cx[i] = static_cast<int>((pc.x[i] - min_x) * inv_r);
        pt_cy[i] = static_cast<int>((pc.y[i] - min_y) * inv_r);
        pt_cz[i] = static_cast<int>((pc.z[i] - min_z) * inv_r);
    }

    // Hash table: next power-of-2 above 2*n for cheap bitwise masking.
    int ht_size = 1;
    while (ht_size < 2 * n) ht_size <<= 1;
    const int ht_mask = ht_size - 1;

    // Build chained hash table on the host.
    std::vector<int> ht_head(ht_size, -1);
    std::vector<int> ht_next(n, -1);
    for (int i = 0; i < n; ++i) {
        int h = (int)(spatial_hash(pt_cx[i], pt_cy[i], pt_cz[i]) & (unsigned int)ht_mask);
        ht_next[i]  = ht_head[h];
        ht_head[h]  = i;
    }

    // max_radius: design ensures ~27 cells hold k pts, radius=1 → 3^3=27 cells.
    float cells_needed = static_cast<float>(k) / pts_per_cell;
    int max_radius = static_cast<int>(std::cbrt(cells_needed) / 2.0f) + 3;
    max_radius = std::max(max_radius, 1);

    DeviceBuffer<int> d_x(n), d_y(n), d_z(n), d_I(n), d_out(n);
    DeviceBuffer<int> d_pt_cx(n), d_pt_cy(n), d_pt_cz(n);
    DeviceBuffer<int> d_ht_head(ht_size), d_ht_next(n);

    d_x.upload(pc.x);      d_y.upload(pc.y);      d_z.upload(pc.z);
    d_I.upload(pc.intensity);
    d_pt_cx.upload(pt_cx); d_pt_cy.upload(pt_cy); d_pt_cz.upload(pt_cz);
    d_ht_head.upload(ht_head);
    d_ht_next.upload(ht_next);

    approx_knn_hash_kernel<<<(n + BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        d_x.ptr, d_y.ptr, d_z.ptr, d_I.ptr,
        d_pt_cx.ptr, d_pt_cy.ptr, d_pt_cz.ptr,
        d_ht_head.ptr, d_ht_next.ptr, d_out.ptr,
        ht_mask, n, k, max_radius);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> result(n);
    d_out.download(result);
    return result;
}

// ---------------------------------------------------------------------------
//  Dispatcher: selected at compile time by USE_SPATIAL_HASH_APPROX.
// ---------------------------------------------------------------------------
std::vector<int> run_approx_knn(const PointCloud &pc)
{
#if USE_SPATIAL_HASH_APPROX
    return run_approx_knn_hash(pc);
#else
    return run_approx_knn_grid(pc);
#endif
}

// ===========================================================================
//  3. K-MEANS CLUSTERING
//
//  Optimizations:
//    - Centroids loaded into shared memory per block (reduces global mem reads)
//    - GPU-side convergence flag via atomicExch (avoids full assignment download)
//    - accum and update launched back-to-back with single sync
//    - unsigned long long sums with INT_MAX shift for safe negative coord handling
// ===========================================================================
__global__
void kmeans_assign_kernel(const int   * __restrict__ xs,
                          const int   * __restrict__ ys,
                          const int   * __restrict__ zs,
                          const int   * __restrict__ cx,
                          const int   * __restrict__ cy,
                          const int   * __restrict__ cz,
                          int         * __restrict__ assign,
                          int         * __restrict__ d_changed,
                          int n, int k)
{
    extern __shared__ int s_cent[];
    int* scx = s_cent;
    int* scy = s_cent + k;
    int* scz = s_cent + 2*k;

    for (int c = threadIdx.x; c < k; c += blockDim.x) {
        scx[c] = cx[c]; scy[c] = cy[c]; scz[c] = cz[c];
    }
    __syncthreads();

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    long long best = MY_LLONG_MAX; int bi = 0;
    for (int c = 0; c < k; ++c) {
        long long dx = scx[c]-xs[i], dy = scy[c]-ys[i], dz = scz[c]-zs[i];
        long long d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < best ||
            (d2 == best && (scx[c] < scx[bi] ||
                           (scx[c] == scx[bi] && scy[c] < scy[bi]) ||
                           (scx[c] == scx[bi] && scy[c] == scy[bi] && scz[c] < scz[bi])))) {
            best = d2; bi = c;
        }
    }
    if (assign[i] != bi) atomicExch(d_changed, 1);
    assign[i] = bi;
}

__global__
void kmeans_accum_kernel(const int   * __restrict__ xs,
                         const int   * __restrict__ ys,
                         const int   * __restrict__ zs,
                         const int   * __restrict__ assign,
                         unsigned long long *sum_x,
                         unsigned long long *sum_y,
                         unsigned long long *sum_z,
                         int *cnt, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int c = assign[i];
    // Shift by INT_MAX+1 to make all coords non-negative before unsigned accumulation.
    // Safe for any int coordinate: shifted range = [0, 2*INT_MAX+1] = [0, UINT_MAX]
    atomicAdd(&sum_x[c], (unsigned long long)((long long)xs[i] + (long long)INT_MAX + 1LL));
    atomicAdd(&sum_y[c], (unsigned long long)((long long)ys[i] + (long long)INT_MAX + 1LL));
    atomicAdd(&sum_z[c], (unsigned long long)((long long)zs[i] + (long long)INT_MAX + 1LL));
    atomicAdd(&cnt[c], 1);
}

__global__
void kmeans_update_kernel(int *cx, int *cy, int *cz,
                          const unsigned long long *sum_x,
                          const unsigned long long *sum_y,
                          const unsigned long long *sum_z,
                          const int *cnt, int k)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k || cnt[c] == 0) return;
    // Divide first (unsigned), then shift back to signed
    cx[c] = static_cast<int>((long long)(sum_x[c] / cnt[c]) - (long long)INT_MAX - 1LL);
    cy[c] = static_cast<int>((long long)(sum_y[c] / cnt[c]) - (long long)INT_MAX - 1LL);
    cz[c] = static_cast<int>((long long)(sum_z[c] / cnt[c]) - (long long)INT_MAX - 1LL);
}

__global__
void kmeans_histogram_kernel(const int * __restrict__ intensity,
                             const int * __restrict__ assign,
                             int       * __restrict__ hist,
                             int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    atomicAdd(&hist[assign[i] * INTENSITY_LEVELS + intensity[i]], 1);
}

__global__
void kmeans_equalize_kernel(const int * __restrict__ intensity,
                            const int * __restrict__ assign,
                            const int * __restrict__ hist,
                            const int * __restrict__ clust_sz,
                            int       * __restrict__ out,
                            int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int *h = hist + assign[i] * INTENSITY_LEVELS;
    out[i] = equalize_point(h, intensity[i], clust_sz[assign[i]]);
}

std::vector<int> run_kmeans(const PointCloud &pc)
{
    const int n = pc.n, k = pc.k, T = pc.T;

    std::vector<int> h_cx(k), h_cy(k), h_cz(k);
    for (int c = 0; c < k; ++c) {
        h_cx[c] = pc.x[c]; h_cy[c] = pc.y[c]; h_cz[c] = pc.z[c];
    }

    DeviceBuffer<int>                d_x(n), d_y(n), d_z(n), d_I(n);
    DeviceBuffer<int>                d_cx(k), d_cy(k), d_cz(k);
    DeviceBuffer<unsigned long long> d_sx(k), d_sy(k), d_sz(k);
    DeviceBuffer<int>                d_cnt(k), d_assign(n);
    DeviceBuffer<int>                d_hist(k * INTENSITY_LEVELS), d_csz(k), d_out(n);
    DeviceBuffer<int>                d_changed(1);

    d_x.upload(pc.x); d_y.upload(pc.y); d_z.upload(pc.z); d_I.upload(pc.intensity);
    d_cx.upload(h_cx); d_cy.upload(h_cy); d_cz.upload(h_cz);
    CUDA_CHECK(cudaMemset(d_assign.ptr, -1, n * sizeof(int)));

    const int blk_n     = (n + BLOCK_SIZE-1) / BLOCK_SIZE;
    const int blk_k     = (k + BLOCK_SIZE-1) / BLOCK_SIZE;
    const int smem_bytes = 3 * k * sizeof(int);

    for (int iter = 0; iter < T; ++iter) {
        d_changed.zero();
        kmeans_assign_kernel<<<blk_n, BLOCK_SIZE, smem_bytes>>>(
            d_x.ptr, d_y.ptr, d_z.ptr,
            d_cx.ptr, d_cy.ptr, d_cz.ptr,
            d_assign.ptr, d_changed.ptr, n, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        if (d_changed.download_one() == 0) break;

        d_sx.zero(); d_sy.zero(); d_sz.zero(); d_cnt.zero();
        kmeans_accum_kernel<<<blk_n, BLOCK_SIZE>>>(
            d_x.ptr, d_y.ptr, d_z.ptr, d_assign.ptr,
            d_sx.ptr, d_sy.ptr, d_sz.ptr, d_cnt.ptr, n);
        CUDA_CHECK(cudaGetLastError());
        kmeans_update_kernel<<<blk_k, BLOCK_SIZE>>>(
            d_cx.ptr, d_cy.ptr, d_cz.ptr,
            d_sx.ptr, d_sy.ptr, d_sz.ptr, d_cnt.ptr, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    d_hist.zero();
    kmeans_histogram_kernel<<<blk_n, BLOCK_SIZE>>>(d_I.ptr, d_assign.ptr, d_hist.ptr, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Recount cluster sizes — d_cnt may be stale if converged early
    d_cnt.zero(); d_sx.zero(); d_sy.zero(); d_sz.zero();
    kmeans_accum_kernel<<<blk_n, BLOCK_SIZE>>>(
        d_x.ptr, d_y.ptr, d_z.ptr, d_assign.ptr,
        d_sx.ptr, d_sy.ptr, d_sz.ptr, d_cnt.ptr, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    d_csz.copy_from(d_cnt);

    kmeans_equalize_kernel<<<blk_n, BLOCK_SIZE>>>(
        d_I.ptr, d_assign.ptr, d_hist.ptr, d_csz.ptr, d_out.ptr, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> result(n);
    d_out.download(result);
    return result;
}

// ===========================================================================
//  main
// ===========================================================================
int main(int argc, char *argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>\n";
        return 1;
    }
    try {
        PointCloud pc = read_input(argv[1]);
        std::cout << "N=" << pc.n << "  k=" << pc.k << "  T=" << pc.T << '\n';
        {
            double t0 = omp_get_wtime();
            write_output("knn.txt", pc, run_knn(pc));
            std::cout << "KNN:        " << omp_get_wtime() - t0 << " s\n";
        }
        {
            double t0 = omp_get_wtime();
            write_output("approx_knn.txt", pc, run_approx_knn(pc));
            std::cout << "Approx KNN: " << omp_get_wtime() - t0 << " s\n";
        }
        {
            double t0 = omp_get_wtime();
            write_output("kmeans.txt", pc, run_kmeans(pc));
            std::cout << "K-Means:    " << omp_get_wtime() - t0 << " s\n";
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
