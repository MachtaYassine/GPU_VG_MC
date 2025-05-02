/**************************************************************
Variance Gamma Nested Monte Carlo Simulation
Inspired by Lokman A. Abbas-Turki's Exponential OU code
***************************************************************/
#include <stdio.h>
#include <curand_kernel.h>
#include <math.h>
#include "VG_cuda_algorithms.cu"
#include <sys/stat.h>
#include <sys/types.h>
#include <omp.h>
#include <string.h>



// Function that catches the error 
void testCUDA(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        printf("There is an error in file %s at line %d\n", file, line);
        exit(EXIT_FAILURE);
    }
}
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

// Kernel to initialize curand states
__global__ void init_curand_states(curandState_t* states, int n, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) curand_init(seed, idx, 0, &states[idx]);
}

// Set the state for each thread
__global__ void init_curand_state_k(curandState* state) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    curand_init(0, idx, 0, &state[idx]);
}

/**
 * @brief Monte Carlo kernel for the Variance Gamma (VG) model using index decoding for parameter grid.
 *
 * Each thread simulates option pricing for a unique combination of parameters (T, K, kappa, theta, sigma),
 * using a flat thread index and decoding it into multidimensional parameter indices, as in MCexpOU.cu.
 * This approach enables efficient parallelization over all parameter combinations.
 */
// __global__ void vg_mc_kernel_flat_indexed(
//     int nT, int nKappa, int nTheta, int nSigma, int nK,
//     const float* T_grid,
//     const float* kappa_grid,
//     const float* theta_grid,
//     const float* sigma_grid,
//     const float* K_grid, // flattened: nT * nK
//     int nSteps,
//     int nPaths,
//     float* d_price,
//     float* d_var,
//     curandState_t* states
// ) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int total = nT * nKappa * nTheta * nSigma * nK;
//     if (idx >= total) return;
//     int same = idx;
//     int isigma = same % nSigma;
//     same /= nSigma;
//     int ik = same % nK;
//     same /= nK;
//     int itheta = same % nTheta;
//     same /= nTheta;
//     int ikappa = same % nKappa;
//     same /= nKappa;
//     int it = same % nT;
//     // Serpentine (zigzag) ordering
//     if (ik % 2 == 1) isigma = nSigma - 1 - isigma;
//     if (itheta % 2 == 1) ik = nK - 1 - ik;
//     if (ikappa % 2 == 1) itheta = nTheta - 1 - itheta;
//     if (it % 2 == 1) ikappa = nKappa - 1 - ikappa;
//     // Fetch parameters
//     float T = T_grid[it];
//     float kappa = kappa_grid[ikappa];
//     float theta = theta_grid[itheta];
//     float sigma = sigma_grid[isigma];
//     float K = K_grid[it * nK + ik];
//     float sum = 0.0f, sum2 = 0.0f;
//     curandState_t localState = states[idx];
//     for (int i = 0; i < nPaths; ++i) {
//         float XVG = simulate_vg(T, nSteps, sigma, theta, kappa, &localState);
//         float YT = expf(XVG); // Y0 = 1
//         float payoff = fmaxf(YT - K, 0.0f);
//         sum += payoff;
//         sum2 += payoff * payoff;
//     }
//     d_price[idx] = sum / nPaths;
//     d_var[idx] = sum2 / nPaths;
//     states[idx] = localState;
// }

/**
 * @brief Monte Carlo kernel for the Variance Gamma (VG) model for a single (T, K) combination.
 *
 * Each thread simulates option pricing for a unique combination of parameters (kappa, theta, sigma),
 * using a flat thread index and decoding it into multidimensional parameter indices.
 * This approach enables efficient parallelization over all parameter combinations for a single (T, K).
 */
__device__ float kappa_d[10];
__device__ float theta_d[10];
__device__ float sigma_d[10];

__global__ void vg_mc_kernel_flat_indexed_singleT(
    float T,
    const float* K_grid,
    int nK,
    int nSteps,
    int nPaths,
    float* d_price,
    float* d_var,
    curandState_t* states,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int same = idx;
    int isigma = same % 10;
    same /= 10;
    int itheta = same % 10;
    same /= 10;
    int ikappa = same % 10;
    same /= 10;
    int ik = same % nK;
    // Fetch parameters from device arrays
    float kappa = kappa_d[ikappa];
    float theta = theta_d[itheta];
    float sigma = sigma_d[isigma];
    float K = K_grid[ik];
    float sum = 0.0f, sum2 = 0.0f;
    curandState_t localState = states[idx];
    for (int i = 0; i < nPaths; ++i) {
        float XVG = simulate_vg(T, nSteps, sigma, theta, kappa, &localState);
        float YT = expf(XVG); // Y0 = 1
        float payoff = fmaxf(YT - K, 0.0f);
        sum += payoff;
        sum2 += payoff * payoff;
    }
    d_price[idx] = sum / nPaths;
    d_var[idx] = sum2 / nPaths;
    states[idx] = localState;
}

// Row-major (OU-style) indexing kernel
__global__ void vg_mc_kernel_rowmajor_dummy(
    float T, float K,
    const float* kappa_grid,
    const float* theta_grid,
    const float* sigma_grid,
    int nKappa, int nTheta, int nSigma,
    int nSteps, int nPaths,
    curandState_t* states,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int same = idx;
    int isigma = same % nSigma;
    same /= nSigma;
    int itheta = same % nTheta;
    same /= nTheta;
    int ikappa = same % nKappa;
    // Fetch parameters
    float kappa = kappa_grid[ikappa];
    float theta = theta_grid[itheta];
    float sigma = sigma_grid[isigma];
    curandState_t localState = states[idx];
    float dummy = 0.0f;
    for (int i = 0; i < nPaths; ++i) {
        float XVG = simulate_vg(T, nSteps, sigma, theta, kappa, &localState);
        float YT = expf(XVG);
        float payoff = fmaxf(YT - K, 0.0f);
        dummy += payoff;
    }
    states[idx] = localState;
}

// Serpentine indexing kernel
__global__ void vg_mc_kernel_serpentine_dummy(
    float T, float K,
    const float* kappa_grid,
    const float* theta_grid,
    const float* sigma_grid,
    int nKappa, int nTheta, int nSigma,
    int nSteps, int nPaths,
    curandState_t* states,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int same = idx;
    int isigma = same % nSigma;
    same /= nSigma;
    int itheta = same % nTheta;
    same /= nTheta;
    int ikappa = same % nKappa;
    // Serpentine: reverse kappa if K is odd, reverse sigma if theta is odd
    if (((int)(K * 1000)) % 2 == 1) ikappa = nKappa - 1 - ikappa; // K is float, so use int cast
    if (itheta % 2 == 1) isigma = nSigma - 1 - isigma;
    float kappa = kappa_grid[ikappa];
    float theta = theta_grid[itheta];
    float sigma = sigma_grid[isigma];
    curandState_t localState = states[idx];
    float dummy = 0.0f;
    for (int i = 0; i < nPaths; ++i) {
        float XVG = simulate_vg(T, nSteps, sigma, theta, kappa, &localState);
        float YT = expf(XVG);
        float payoff = fmaxf(YT - K, 0.0f);
        dummy += payoff;
    }
    states[idx] = localState;
}

// Dummy timing function to compare both kernels
void benchmark_indexing_kernels(
    float T, float K,
    const float* kappa_grid, const float* theta_grid, const float* sigma_grid,
    int nKappa, int nTheta, int nSigma, int nSteps, int nPaths, int nRepeat
) {
    int total = nKappa * nTheta * nSigma;
    float *d_kappa, *d_theta, *d_sigma;
    testCUDA(cudaMalloc(&d_kappa, total * sizeof(float)));
    testCUDA(cudaMalloc(&d_theta, total * sizeof(float)));
    testCUDA(cudaMalloc(&d_sigma, total * sizeof(float)));
    float *h_kappa = (float*)malloc(total * sizeof(float));
    float *h_theta = (float*)malloc(total * sizeof(float));
    float *h_sigma = (float*)malloc(total * sizeof(float));
    for (int i = 0, ikappa = 0; ikappa < nKappa; ++ikappa)
        for (int itheta = 0; itheta < nTheta; ++itheta)
            for (int isigma = 0; isigma < nSigma; ++isigma, ++i) {
                h_kappa[i] = kappa_grid[ikappa];
                h_theta[i] = theta_grid[itheta];
                h_sigma[i] = sigma_grid[isigma];
            }
    testCUDA(cudaMemcpy(d_kappa, h_kappa, total * sizeof(float), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(d_theta, h_theta, total * sizeof(float), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(d_sigma, h_sigma, total * sizeof(float), cudaMemcpyHostToDevice));
    curandState_t* d_states;
    testCUDA(cudaMalloc(&d_states, total * sizeof(curandState_t)));
    int threads = 1024;
    int blocks = (total + threads - 1) / threads;
    init_curand_states<<<blocks, threads>>>(d_states, total, 42UL);
    testCUDA(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    float ms, sum_row = 0.0f, sum_serp = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Row-major timing
    for (int rep = 0; rep < nRepeat; ++rep) {
        cudaEventRecord(start);
        vg_mc_kernel_rowmajor_dummy<<<blocks, threads>>>(T, K, d_kappa, d_theta, d_sigma, nKappa, nTheta, nSigma, nSteps, nPaths, d_states, total);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        sum_row += ms;
    }
    printf("Row-major kernel (avg of %d runs): %f ms\n", nRepeat, sum_row / nRepeat);
    // Serpentine timing
    for (int rep = 0; rep < nRepeat; ++rep) {
        cudaEventRecord(start);
        vg_mc_kernel_serpentine_dummy<<<blocks, threads>>>(T, K, d_kappa, d_theta, d_sigma, nKappa, nTheta, nSigma, nSteps, nPaths, d_states, total);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        sum_serp += ms;
    }
    printf("Serpentine kernel (avg of %d runs): %f ms\n", nRepeat, sum_serp / nRepeat);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_kappa); free(h_theta); free(h_sigma);
    testCUDA(cudaFree(d_kappa)); testCUDA(cudaFree(d_theta)); testCUDA(cudaFree(d_sigma));
    testCUDA(cudaFree(d_states));
}

// Host function to generate strike grid
void strikeInterval(float* K, float T) {
    float fidx = T * 12.0f + 1.0f;
    int i = 0;
    float coef = 1.0f;
    float delta;
    while (i < fidx) {
        coef *= (1.02f);
        i++;
    }
    delta = pow(coef, 1.0f / 8.0f);
    K[15] = coef;
    for (i = 1; i < 16; i++) {
        K[15 - i] = K[15 - i + 1] / delta;
    }
}

int main(int argc, char** argv) {
    int nT = 4;
    int nK = 16;
    int nKappa = 10, nTheta = 10, nSigma = 10;
    int nSteps = 64;
    int nPaths = 8192;
    int nRepeat = 5;
    // Fixed parameter grids
    float sigma_grid[10] = { 0.1f, 0.12f, 0.13f, 0.14f, 0.15f, 0.16f, 0.17f, 0.18f, 0.19f, 0.2f };
    float theta_grid[10] = { -0.34f, -0.3f, -0.27f, -0.24f, -0.21f, -0.25f, -0.26f, -0.35f, -0.4f, -0.45f };
    float kappa_grid[10] = { 0.11f, 0.12f, 0.13f, 0.14f, 0.15f, 0.16f, 0.17f, 0.18f, 0.19f, 0.20f };
    // Update T_grid to increment from 0.125 to 1.0 (8 values)
    float T_grid[4] = { 0.25f, 0.5f, 0.75f, 1.0f};
    // Check for benchmarking flag
    int do_benchmark = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--benchmark-indexing") == 0) {
            do_benchmark = 1;
        }
    }
    if (do_benchmark) {
        printf("\nBenchmarking indexing kernels for T=%.2f, K=%.4f\n", T_grid[0], 1.0f);
        benchmark_indexing_kernels(
            T_grid[0],           // T
            1.0f,                // K (or any K value you want to test)
            kappa_grid, theta_grid, sigma_grid,
            nKappa, nTheta, nSigma, nSteps, nPaths, nRepeat
        );
        return 0;
    }

    mkdir("Training", 0777);
    mkdir("Testing", 0777);
    char fname[256];
    for (int it = 0; it < nT; ++it) {
        float T = T_grid[it];
        float Kvec[16];
        strikeInterval(Kvec, T);
        int nK = 16;
        int total = nKappa * nTheta * nSigma * nK;
        float *d_price, *d_var;
        testCUDA(cudaMalloc(&d_price, total * sizeof(float)));
        testCUDA(cudaMalloc(&d_var, total * sizeof(float)));
        curandState_t* d_states;
        testCUDA(cudaMalloc(&d_states, total * sizeof(curandState_t)));
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        init_curand_states<<<blocks, threads>>>(d_states, total, 42UL);
        testCUDA(cudaDeviceSynchronize());
        float* d_Kvec;
        testCUDA(cudaMalloc(&d_Kvec, nK * sizeof(float)));
        testCUDA(cudaMemcpyToSymbol(kappa_d, kappa_grid, nKappa * sizeof(float)));
        testCUDA(cudaMemcpyToSymbol(theta_d, theta_grid, nTheta * sizeof(float)));
        testCUDA(cudaMemcpyToSymbol(sigma_d, sigma_grid, nSigma * sizeof(float)));
        testCUDA(cudaMemcpy(d_Kvec, Kvec, nK * sizeof(float), cudaMemcpyHostToDevice));
        vg_mc_kernel_flat_indexed_singleT<<<blocks, threads>>>(T, d_Kvec, nK, nSteps, nPaths, d_price, d_var, d_states, total);
        testCUDA(cudaDeviceSynchronize());
        float* h_price = (float*)malloc(total * sizeof(float));
        float* h_var = (float*)malloc(total * sizeof(float));
        testCUDA(cudaMemcpy(h_price, d_price, total * sizeof(float), cudaMemcpyDeviceToHost));
        testCUDA(cudaMemcpy(h_var, d_var, total * sizeof(float), cudaMemcpyDeviceToHost));
        // Write CSV for each K
        for (int ik = 0; ik < nK; ++ik) {
            float K = Kvec[ik];
            const char* folder = (((it * nK + ik) % 10) < 8) ? "Training" : "Testing";
            char fname[256];
            sprintf(fname, "%s/VG_T%.2f_K%.4f.csv", folder, T, K);
            FILE* f = fopen(fname, "w");
            fprintf(f, "kappa,theta,sigma,price,95CI\n");
            for (int ikappa = 0; ikappa < nKappa; ++ikappa)
                for (int itheta = 0; itheta < nTheta; ++itheta)
                    for (int isigma = 0; isigma < nSigma; ++isigma) {
                        int idx = (((ikappa * nTheta + itheta) * nSigma + isigma) * nK) + ik;
                        float price = h_price[idx];
                        float var = h_var[idx] - price * price;
                        float ci = 1.96f * sqrtf(var / nPaths);
                        fprintf(f, "%f,%f,%f,%f,%f\n", kappa_grid[ikappa], theta_grid[itheta], sigma_grid[isigma], price, ci);
                    }
            fclose(f);
        }
        free(h_price); free(h_var);
        testCUDA(cudaFree(d_price)); testCUDA(cudaFree(d_var)); testCUDA(cudaFree(d_states)); testCUDA(cudaFree(d_Kvec));
    }
    return 0;
}
