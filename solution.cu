#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <climits>
#include <cfloat>
#include <cmath>

#include <omp.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ---------------------------------------------------------------------------
//  CUDA error helper
// ---------------------------------------------------------------------------
inline void cuda_check(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("CUDA error at ") + file + ":" + std::to_string(line)
            + " - " + cudaGetErrorString(err));
}
#define CUDA_CHECK(call) cuda_check((call), __FILE__, __LINE__)

template <typename T>
struct DeviceBuffer {
    T   *ptr  = nullptr;
    int  size = 0;

    explicit DeviceBuffer(int n) : size(n) {
        CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    }
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }

    DeviceBuffer(const DeviceBuffer &)            = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    DeviceBuffer(DeviceBuffer &&o) noexcept : ptr(o.ptr), size(o.size) {
        o.ptr = nullptr;
    }

    void upload(const std::vector<T> &src) const {
        CUDA_CHECK(cudaMemcpy(ptr, src.data(), size * sizeof(T),
                              cudaMemcpyHostToDevice));
    }
    void download(std::vector<T> &dst) const {
        CUDA_CHECK(cudaMemcpy(dst.data(), ptr, size * sizeof(T),
                              cudaMemcpyDeviceToHost));
    }
    void zero() const { CUDA_CHECK(cudaMemset(ptr, 0, size * sizeof(T))); }
    void copy_from(const DeviceBuffer<T> &src) const {
        CUDA_CHECK(cudaMemcpy(ptr, src.ptr, size * sizeof(T),
                              cudaMemcpyDeviceToDevice));
    }
};




constexpr int INTENSITY_LEVELS = 256;
constexpr int BLOCK_SIZE       = 256;
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


__device__ __forceinline__
int equalize_point(const int * __restrict__ hist, int orig, int m)
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


__device__ __forceinline__
void heap_bubble_up(long long *dist, int *idx, int pos)
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

__device__ __forceinline__
void heap_sift_down(long long *dist, int *idx, int k)
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


#define HEAP_INSERT(dist, idx, heap_size, k, d2, j)    
    if ((heap_size) < (k)) {                            
        (dist)[(heap_size)] = (d2);                     
        (idx) [(heap_size)] = (j);                      
        heap_bubble_up((dist), (idx), (heap_size));     
        (heap_size)++;                                  
    } else if ((d2) < (dist)[0]) {                      
        (dist)[0] = (d2); (idx)[0] = (j);              
        heap_sift_down((dist), (idx), (k));             
    }


    
__global__
void knn_kernel(const int * __restrict__ xs,
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


// __global__
// void approx_knn_kernel(const int * __restrict__ xs,
//                        const int * __restrict__ ys,
//                        const int * __restrict__ zs,
//                        const int * __restrict__ intensity,
//                        const int * __restrict__ sorted_ids,
//                        const int * __restrict__ cell_start,
//                        const int * __restrict__ cell_end,
//                        int       * __restrict__ out,
//                        int min_x, int min_y, int min_z,
//                        float inv_r,
//                        int nx, int ny, int nz,
//                        int n, int k)
// {
//     const int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i >= n) return;
//     if (k == 0) { out[i] = intensity[i]; return; }

//     const int qx = xs[i], qy = ys[i], qz = zs[i];
//     const int cx = min(static_cast<int>((qx - min_x) * inv_r), nx-1);
//     const int cy = min(static_cast<int>((qy - min_y) * inv_r), ny-1);
//     const int cz = min(static_cast<int>((qz - min_z) * inv_r), nz-1);

//     long long heap_dist[MAX_K];
//     int       heap_idx [MAX_K];
//     int       heap_size = 0;

//     for (int dz = -1; dz <= 1; ++dz) {
//         int nz_ = cz + dz; if (nz_ < 0 || nz_ >= nz) continue;
//         for (int dy = -1; dy <= 1; ++dy) {
//             int ny_ = cy + dy; if (ny_ < 0 || ny_ >= ny) continue;
//             for (int dx = -1; dx <= 1; ++dx) {
//                 int nx_ = cx + dx; if (nx_ < 0 || nx_ >= nx) continue;
//                 const int cell = nz_ * ny * nx + ny_ * nx + nx_;
//                 for (int s = cell_start[cell]; s < cell_end[cell]; ++s) {
//                     const int j = sorted_ids[s];
//                     if (j == i) continue;
//                     long long ddx = xs[j]-qx, ddy = ys[j]-qy, ddz = zs[j]-qz;
//                     long long d2  = ddx*ddx + ddy*ddy + ddz*ddz;
//                     HEAP_INSERT(heap_dist, heap_idx, heap_size, k, d2, j);
//                 }
//             }
//         }
//     }

//     int hist[INTENSITY_LEVELS] = {};
//     hist[intensity[i]]++;
//     for (int t = 0; t < heap_size; ++t) hist[intensity[heap_idx[t]]]++;

//     out[i] = equalize_point(hist, intensity[i], heap_size + 1);
// }



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
    for (int i = 0; i < n; ++i) ++count[cell_id[i]];

    std::vector<int> cell_start(total_cells), cell_end(total_cells);
    cell_start[0] = 0;
    for (int c = 1; c < total_cells; ++c)
        cell_start[c] = cell_start[c-1] + count[c-1];
    for (int c = 0; c < total_cells; ++c)
        cell_end[c] = cell_start[c] + count[c];

    std::vector<int> sorted_ids(n), tmp(total_cells, 0);
    for (int i = 0; i < n; ++i)
        sorted_ids[cell_start[cell_id[i]] + tmp[cell_id[i]]++] = i;

    DeviceBuffer<int> d_x(n), d_y(n), d_z(n), d_I(n), d_out(n);
    DeviceBuffer<int> d_sorted(n), d_cs(total_cells), d_ce(total_cells);
    d_x.upload(pc.x); d_y.upload(pc.y); d_z.upload(pc.z); d_I.upload(pc.intensity);
    d_sorted.upload(sorted_ids);
    d_cs.upload(cell_start); d_ce.upload(cell_end);

    approx_knn_kernel<<<(n + BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
        d_x.ptr, d_y.ptr, d_z.ptr, d_I.ptr,
        d_sorted.ptr, d_cs.ptr, d_ce.ptr, d_out.ptr,
        min_x, min_y, min_z, inv_r, nx, ny, nz, n, k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> result(n);
    d_out.download(result);
    return result;
}


__global__
void kmeans_assign_kernel(const int   * __restrict__ xs,
                          const int   * __restrict__ ys,
                          const int   * __restrict__ zs,
                          const float * __restrict__ cx,
                          const float * __restrict__ cy,
                          const float * __restrict__ cz,
                          int         * __restrict__ assign,
                          int n, int k)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float px = xs[i], py = ys[i], pz = zs[i];
    float best = FLT_MAX; int bi = 0;
    for (int c = 0; c < k; ++c) {
        float dx = cx[c]-px, dy = cy[c]-py, dz = cz[c]-pz;
        float d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < best) { best = d2; bi = c; }
    }
    assign[i] = bi;
}

__global__
void kmeans_accum_kernel(const int   * __restrict__ xs,
                         const int   * __restrict__ ys,
                         const int   * __restrict__ zs,
                         const int   * __restrict__ assign,
                         float *sum_x, float *sum_y, float *sum_z,
                         int   *cnt, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int c = assign[i];
    atomicAdd(&sum_x[c], static_cast<float>(xs[i]));
    atomicAdd(&sum_y[c], static_cast<float>(ys[i]));
    atomicAdd(&sum_z[c], static_cast<float>(zs[i]));
    atomicAdd(&cnt[c], 1);
}

__global__
void kmeans_update_kernel(float       *cx, float       *cy, float       *cz,
                          const float *sum_x, const float *sum_y, const float *sum_z,
                          const int   *cnt, int k)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k || cnt[c] == 0) return;
    const float inv = 1.0f / static_cast<float>(cnt[c]);
    cx[c] = sum_x[c] * inv;
    cy[c] = sum_y[c] * inv;
    cz[c] = sum_z[c] * inv;
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

    std::vector<float> h_cx(k), h_cy(k), h_cz(k);
    for (int c = 0; c < k; ++c) {
        h_cx[c] = static_cast<float>(pc.x[c]);
        h_cy[c] = static_cast<float>(pc.y[c]);
        h_cz[c] = static_cast<float>(pc.z[c]);
    }

    DeviceBuffer<int>   d_x(n), d_y(n), d_z(n), d_I(n);
    DeviceBuffer<float> d_cx(k), d_cy(k), d_cz(k);
    DeviceBuffer<float> d_sx(k), d_sy(k), d_sz(k);
    DeviceBuffer<int>   d_cnt(k), d_assign(n);
    DeviceBuffer<int>   d_hist(k * INTENSITY_LEVELS), d_csz(k), d_out(n);

    d_x.upload(pc.x); d_y.upload(pc.y); d_z.upload(pc.z); d_I.upload(pc.intensity);
    d_cx.upload(h_cx); d_cy.upload(h_cy); d_cz.upload(h_cz);

    const int blk_n = (n + BLOCK_SIZE-1) / BLOCK_SIZE;
    const int blk_k = (k + BLOCK_SIZE-1) / BLOCK_SIZE;

    std::vector<int> h_assign(n), h_assign_prev(n, -1);

    for (int iter = 0; iter < T; ++iter) {
        kmeans_assign_kernel<<<blk_n, BLOCK_SIZE>>>(
            d_x.ptr, d_y.ptr, d_z.ptr,
            d_cx.ptr, d_cy.ptr, d_cz.ptr,
            d_assign.ptr, n, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        d_assign.download(h_assign);
        int changed = 0;
        #pragma omp parallel for schedule(static) reduction(+: changed)
        for (int i = 0; i < n; ++i)
            if (h_assign[i] != h_assign_prev[i]) ++changed;
        if (changed == 0) break;
        h_assign_prev = h_assign;

        d_sx.zero(); d_sy.zero(); d_sz.zero(); d_cnt.zero();
        kmeans_accum_kernel<<<blk_n, BLOCK_SIZE>>>(
            d_x.ptr, d_y.ptr, d_z.ptr, d_assign.ptr,
            d_sx.ptr, d_sy.ptr, d_sz.ptr, d_cnt.ptr, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        kmeans_update_kernel<<<blk_k, BLOCK_SIZE>>>(
            d_cx.ptr, d_cy.ptr, d_cz.ptr,
            d_sx.ptr, d_sy.ptr, d_sz.ptr, d_cnt.ptr, k);
        CUDA_CHECK(cudaGetLastError());
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
        // No structured bindings -- not supported by this nvcc version
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

