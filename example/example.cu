#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>


// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void vectorAdd(const float* __restrict__ a,
                          const float* __restrict__ b,
                          float* __restrict__ out,
                          size_t n)
{
    size_t idx = threadIdx.x + blockIdx.x * (size_t)blockDim.x;
    if (idx < n)
        out[idx] = a[idx] + b[idx];
}


// ---------------------------------------------------------------------------
// Error-checking helper
// ---------------------------------------------------------------------------

static void check(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err)
                  << "  (" << file << ":" << line << ")\n";
        std::exit(1);
    }
}
#define CHECK(e) check((e), __FILE__, __LINE__)


// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main()
{
    // ------------------------------------------------------------------
    // Device info
    // ------------------------------------------------------------------
    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "============================================================\n";
    std::cout << "  CUDA Vector-Addition Benchmark\n";
    std::cout << "============================================================\n";
    std::cout << "Device : " << prop.name << "\n";
    (void)prop;

    // ------------------------------------------------------------------
    // Benchmark parameters
    // ------------------------------------------------------------------
    const std::size_t N           = 1u << 25;   // 32 M floats = 128 MB per array
    const std::size_t BYTES       = N * sizeof(float);
    const int         WARMUP      = 20;
    const int         ITERATIONS  = 200;
    const int         BLOCK       = 256;
    const int         GRID        = static_cast<int>((N + BLOCK - 1) / BLOCK);

    std::cout << "Array size : " << N << " float32 elements  ("
              << BYTES / (1024 * 1024) << " MiB each)\n";
    std::cout << "Iterations : " << ITERATIONS
              << "  (+ " << WARMUP << " warm-up)\n\n";

    // ------------------------------------------------------------------
    // Host arrays
    // ------------------------------------------------------------------
    std::vector<float> h_a(N), h_b(N), h_out(N);
    for (std::size_t i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i)     / static_cast<float>(N);
        h_b[i] = static_cast<float>(N - i) / static_cast<float>(N);
    }

    // ------------------------------------------------------------------
    // Device arrays
    // ------------------------------------------------------------------
    float *d_a{}, *d_b{}, *d_out{};
    CHECK(cudaMalloc(&d_a,   BYTES));
    CHECK(cudaMalloc(&d_b,   BYTES));
    CHECK(cudaMalloc(&d_out, BYTES));

    CHECK(cudaMemcpy(d_a, h_a.data(), BYTES, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b.data(), BYTES, cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------
    // Warm-up
    // ------------------------------------------------------------------
    for (int i = 0; i < WARMUP; ++i)
        vectorAdd<<<GRID, BLOCK>>>(d_a, d_b, d_out, N);
    CHECK(cudaDeviceSynchronize());

    // ------------------------------------------------------------------
    // Timed benchmark using CUDA events
    // ------------------------------------------------------------------
    cudaEvent_t t_start, t_stop;
    CHECK(cudaEventCreate(&t_start));
    CHECK(cudaEventCreate(&t_stop));

    CHECK(cudaEventRecord(t_start));
    for (int i = 0; i < ITERATIONS; ++i)
        vectorAdd<<<GRID, BLOCK>>>(d_a, d_b, d_out, N);
    CHECK(cudaEventRecord(t_stop));
    CHECK(cudaEventSynchronize(t_stop));

    float elapsed_ms{};
    CHECK(cudaEventElapsedTime(&elapsed_ms, t_start, t_stop));

    const double avg_ms  = elapsed_ms / ITERATIONS;
    // Memory traffic: 2 reads (a, b) + 1 write (out) = 3 * BYTES
    const double bw_gb_s = (3.0 * static_cast<double>(BYTES)) /
                           (avg_ms * 1e-3) / 1e9;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "------------------------------------------------------------\n";
    std::cout << "Avg kernel time : " << avg_ms  << " ms\n";
    std::cout << std::setprecision(2);
    std::cout << "Effective BW    : " << bw_gb_s << " GB/s\n";
    std::cout << "------------------------------------------------------------\n\n";

    // ------------------------------------------------------------------
    // Correctness check
    // ------------------------------------------------------------------
    CHECK(cudaMemcpy(h_out.data(), d_out, BYTES, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (std::size_t i = 0; i < N; ++i) {
        const float expected = h_a[i] + h_b[i];
        if (std::fabs(h_out[i] - expected) > 1e-5f) {
            std::cerr << "Mismatch at index " << i
                      << ": got " << h_out[i]
                      << ", expected " << expected << "\n";
            ok = false;
            break;
        }
    }
    std::cout << "Correctness check : " << (ok ? "PASS" : "FAIL") << "\n";

    // ------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_out));
    CHECK(cudaEventDestroy(t_start));
    CHECK(cudaEventDestroy(t_stop));

    return ok ? 0 : 1;
}
