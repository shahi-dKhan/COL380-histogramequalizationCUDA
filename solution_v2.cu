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
        if(size != other.size) throw std::runtime_error("Size mismatch in copyFrom");
        CUDA_CHECK(cudaMemcpy(ptr, other.ptr, size * sizeof(T), cudaMemcpyDeviceToDevice));
    }
    T download_one() const {
        T val;
        CUDA_CHECK(cudaMemcpy(&val, ptr, sizeof(T), cudaMemcpyDeviceToHost));
        return val;
    }
};

// Optimization 1 & 2: BLOCK_SIZE reduced to 128
constexpr int INTENSITY_LEVELS = 256;
constexpr int BLOCK_SIZE       = 128;
constexpr int TILE             = 128;  // == BLOCK_SIZE
constexpr int MAX_K            = 128;

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

// Optimization 1 & 2: __launch_bounds__(128, 2) + shared memory tiling
__launch_bounds__(128, 2)
__global__ void knn_kernel(const int * __restrict__ xs,
                           const int * __restrict__ ys,
                           const int * __restrict__ zs,
                           const int * __restrict__ intensity,
                           int       * __restrict__ out,
                           int n, int k)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    // ghost threads participate in __syncthreads inside tile loop; early-exit after
    const bool is_ghost = (i >= n);

    int qx = 0, qy = 0, qz = 0;
    if (!is_ghost) {
        qx = xs[i]; qy = ys[i]; qz = zs[i];
    }

    // k==0: handled after the tile loop (heap stays empty, equalize returns orig)

    long long heap_dist[MAX_K];
    int       heap_idx [MAX_K];
    int       heap_size = 0;

    __shared__ int s_x[TILE], s_y[TILE], s_z[TILE];

    const int num_tiles = (n + TILE - 1) / TILE;
    for (int t = 0; t < num_tiles; ++t) {
        // All threads (including ghosts) collaboratively load one tile
        int src = t * TILE + threadIdx.x;
        s_x[threadIdx.x] = (src < n) ? xs[src] : 0;
        s_y[threadIdx.x] = (src < n) ? ys[src] : 0;
        s_z[threadIdx.x] = (src < n) ? zs[src] : 0;
        __syncthreads();

        if (!is_ghost) {
            int tile_end = min(TILE, n - t * TILE);
            for (int jj = 0; jj < tile_end; ++jj) {
                int j = t * TILE + jj;
                if (j == i) continue;
                long long dx = s_x[jj] - qx;
                long long dy = s_y[jj] - qy;
                long long dz = s_z[jj] - qz;
                long long d2 = dx*dx + dy*dy + dz*dz;
                HEAP_INSERT(heap_dist, heap_idx, heap_size, k, d2, j);
            }
        }
        __syncthreads();
    }

    if (is_ghost) return;

    if (k == 0) { out[i] = intensity[i]; return; }

    int hist[INTENSITY_LEVELS] = {};
    hist[intensity[i]]++;
    for (int t = 0; t < heap_size; ++t) hist[intensity[heap_idx[t]]]++;

    out[i] = equalize_point(hist, intensity[i], heap_size + 1);
}

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

// Optimization 7: max_radius parameter replaces unbounded while loop
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
    // Optimization 7: bounded for loop + __all_sync warp-level early exit
    for (int radius = 1; radius <= max_radius; ++radius) {
        for (int dz = -radius; dz <= radius; ++dz) {
            int nz_ = cz + dz; if (nz_ < 0 || nz_ >= nz) continue;
            for (int dy = -radius; dy <= radius; ++dy) {
                int ny_ = cy + dy; if (ny_ < 0 || ny_ >= ny) continue;
                for (int dx = -radius; dx <= radius; ++dx) {
                    // skip inner cells already searched in previous radius
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
        // Warp-level early exit: break only when the entire active warp has k neighbors
        if (__all_sync(__activemask(), heap_size >= k)) break;
    }

    int hist[INTENSITY_LEVELS] = {};
    hist[intensity[i]]++;
    for (int t = 0; t < heap_size; ++t) hist[intensity[heap_idx[t]]]++;
    out[i] = equalize_point(hist, intensity[i], heap_size + 1);
}

std::vector<int> run_approx_knn(const PointCloud &pc)
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

    // Optimization 4: parallel cell counting
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

    // Optimization 7: compute max_radius on the host
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

// Optimization 5: shared centroids via dynamic shared memory
// Optimization 3: convergence flag atomicExch
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
    // Optimization 5: dynamic shared memory layout: [cx | cy | cz]
    extern __shared__ int s_cent[];
    int* scx = s_cent;
    int* scy = s_cent + k;
    int* scz = s_cent + 2*k;

    // Collaboratively load all k centroids into shared memory
    for (int c = threadIdx.x; c < k; c += blockDim.x) {
        scx[c] = cx[c];
        scy[c] = cy[c];
        scz[c] = cz[c];
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
            best = d2;
            bi = c;
        }
    }

    // Optimization 3: detect convergence on GPU via atomicExch
    if (assign[i] != bi) {
        atomicExch(d_changed, 1);
    }
    assign[i] = bi;
}

__global__
void kmeans_accum_kernel(const int   * __restrict__ xs,
                         const int   * __restrict__ ys,
                         const int   * __restrict__ zs,
                         const int   * __restrict__ assign,
                         unsigned long long *sum_x, unsigned long long *sum_y, unsigned long long *sum_z,
                         int   *cnt, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int c = assign[i];
    atomicAdd(&sum_x[c], (unsigned long long)(xs[i]));
    atomicAdd(&sum_y[c], (unsigned long long)(ys[i]));
    atomicAdd(&sum_z[c], (unsigned long long)(zs[i]));
    atomicAdd(&cnt[c], 1);
}

__global__
void kmeans_update_kernel(int       *cx, int       *cy, int       *cz,
                          const unsigned long long *sum_x, const unsigned long long *sum_y, const unsigned long long *sum_z,
                          const int   *cnt, int k)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k || cnt[c] == 0) return;
    cx[c] = static_cast<int>((long long)sum_x[c] / cnt[c]);
    cy[c] = static_cast<int>((long long)sum_y[c] / cnt[c]);
    cz[c] = static_cast<int>((long long)sum_z[c] / cnt[c]);
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
        h_cx[c] = static_cast<int>(pc.x[c]);
        h_cy[c] = static_cast<int>(pc.y[c]);
        h_cz[c] = static_cast<int>(pc.z[c]);
    }

    DeviceBuffer<int>   d_x(n), d_y(n), d_z(n), d_I(n);
    DeviceBuffer<int> d_cx(k), d_cy(k), d_cz(k);
    DeviceBuffer<unsigned long long> d_sx(k), d_sy(k), d_sz(k);
    DeviceBuffer<int>   d_cnt(k), d_assign(n);
    DeviceBuffer<int>   d_hist(k * INTENSITY_LEVELS), d_csz(k), d_out(n);

    // Optimization 3: GPU-side convergence flag
    DeviceBuffer<int>   d_changed(1);

    d_x.upload(pc.x); d_y.upload(pc.y); d_z.upload(pc.z); d_I.upload(pc.intensity);
    d_cx.upload(h_cx); d_cy.upload(h_cy); d_cz.upload(h_cz);

    // Optimization 3: initialize d_assign to -1 so first iteration always runs
    CUDA_CHECK(cudaMemset(d_assign.ptr, -1, n * sizeof(int)));

    const int blk_n = (n + BLOCK_SIZE-1) / BLOCK_SIZE;
    const int blk_k = (k + BLOCK_SIZE-1) / BLOCK_SIZE;

    // Optimization 5: shared memory size for centroids
    const int smem_bytes = 3 * k * sizeof(int);

    for (int iter = 0; iter < T; ++iter) {
        // Optimization 3: reset flag each iteration
        d_changed.zero();

        kmeans_assign_kernel<<<blk_n, BLOCK_SIZE, smem_bytes>>>(
            d_x.ptr, d_y.ptr, d_z.ptr,
            d_cx.ptr, d_cy.ptr, d_cz.ptr,
            d_assign.ptr, d_changed.ptr, n, k);
        CUDA_CHECK(cudaGetLastError());
        // Optimization 6: one sync after assign (needed for convergence check)
        CUDA_CHECK(cudaDeviceSynchronize());

        // Optimization 3: download only 4 bytes to check convergence
        int h_changed = d_changed.download_one();
        if (h_changed == 0) break;

        d_sx.zero(); d_sy.zero(); d_sz.zero(); d_cnt.zero();

        // Optimization 6: launch accum then update back-to-back, no sync between
        kmeans_accum_kernel<<<blk_n, BLOCK_SIZE>>>(
            d_x.ptr, d_y.ptr, d_z.ptr, d_assign.ptr,
            d_sx.ptr, d_sy.ptr, d_sz.ptr, d_cnt.ptr, n);
        CUDA_CHECK(cudaGetLastError());

        kmeans_update_kernel<<<blk_k, BLOCK_SIZE>>>(
            d_cx.ptr, d_cy.ptr, d_cz.ptr,
            d_sx.ptr, d_sy.ptr, d_sz.ptr, d_cnt.ptr, k);
        CUDA_CHECK(cudaGetLastError());
        // Optimization 6: one sync covers both accum and update
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    d_hist.zero();
    kmeans_histogram_kernel<<<blk_n, BLOCK_SIZE>>>(
        d_I.ptr, d_assign.ptr, d_hist.ptr, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Fresh cluster sizes (d_cnt may be stale if converged early)
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
            write_output("knn_v2.txt", pc, run_knn(pc));
            std::cout << "KNN:        " << omp_get_wtime() - t0 << " s\n";
        }
        {
            double t0 = omp_get_wtime();
            write_output("approx_knn_v2.txt", pc, run_approx_knn(pc));
            std::cout << "Approx KNN: " << omp_get_wtime() - t0 << " s\n";
        }
        {
            double t0 = omp_get_wtime();
            write_output("kmeans_v2.txt", pc, run_kmeans(pc));
            std::cout << "K-Means:    " << omp_get_wtime() - t0 << " s\n";
        }

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
