#include <curand_kernel.h>
#include <stdio.h>
#include "VG_cuda_algorithms.cu"

// Kernel to initialize curand states
__global__ void init_curand_states(curandState_t* states, int n, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curand_init(seed, idx, 0, &states[idx]);
}

// Kernel to test and benchmark gamma_johnk and gamma_best
__global__ void test_gamma_generators(float* out_johnk, float* out_best, int n, float a_johnk, float a_best, curandState_t* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curandState_t localState = states[idx];
    // Test Johnk's generator (a <= 1)
    out_johnk[idx] = gamma_johnk(a_johnk, &localState);
    // Test Best's generator (a >= 1)
    out_best[idx] = gamma_best(a_best, &localState);
}

// Kernel for Johnk only
__global__ void test_gamma_johnk(float* out, int n, float a, curandState_t* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curandState_t localState = states[idx];
    out[idx] = gamma_johnk(a, &localState);
}

// Kernel for Best only
__global__ void test_gamma_best(float* out, int n, float a, curandState_t* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curandState_t localState = states[idx];
    out[idx] = gamma_best(a, &localState);
}

int main() {
    int n = 1 << 20; // 1M samples
    float a_johnk = 0.8f;
    float a_best = 2.0f;
    float *d_johnk, *d_best, *h_johnk, *h_best;
    curandState_t* d_states;
    cudaMalloc(&d_johnk, n * sizeof(float));
    cudaMalloc(&d_best, n * sizeof(float));
    cudaMalloc(&d_states, n * sizeof(curandState_t));
    h_johnk = (float*)malloc(n * sizeof(float));
    h_best = (float*)malloc(n * sizeof(float));

    // Initialize RNG states
    init_curand_states<<<n/256, 256>>>(d_states, n, 1234UL);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Benchmark Johnk and Best together (as before)
    cudaEventRecord(start);
    test_gamma_generators<<<n/256, 256>>>(d_johnk, d_best, n, a_johnk, a_best, d_states);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_both = 0;
    cudaEventElapsedTime(&ms_both, start, stop);

    // Benchmark Johnk only
    cudaEventRecord(start);
    test_gamma_johnk<<<n/256, 256>>>(d_johnk, n, a_johnk, d_states);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_johnk = 0;
    cudaEventElapsedTime(&ms_johnk, start, stop);

    // Benchmark Best only
    cudaEventRecord(start);
    test_gamma_best<<<n/256, 256>>>(d_best, n, a_best, d_states);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_best = 0;
    cudaEventElapsedTime(&ms_best, start, stop);

    cudaMemcpy(h_johnk, d_johnk, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_best, d_best, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print a few results
    printf("First 5 Johnk samples (a=%.2f): ", a_johnk);
    for (int i = 0; i < 5; ++i) printf("%f ", h_johnk[i]);
    printf("\nFirst 5 Best samples (a=%.2f): ", a_best);
    for (int i = 0; i < 5; ++i) printf("%f ", h_best[i]);
    printf("\nTime for %d samples: %f ms\n", n, ms_both);
    printf("Time for Johnk only: %f ms\n", ms_johnk);
    printf("Time for Best only: %f ms\n", ms_best);

    free(h_johnk); free(h_best);
    cudaFree(d_johnk); cudaFree(d_best); cudaFree(d_states);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}