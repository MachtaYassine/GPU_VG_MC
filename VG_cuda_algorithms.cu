// Device function: Johnk's generator for gamma (a <= 1)
__device__ float gamma_johnk(float a, curandState_t* state) {
    float X, Y, U, V, E;
    do {
        U = curand_uniform(state);
        V = curand_uniform(state);
        X = powf(U, 1.0f / a);
        Y = powf(V, 1.0f / (1.0f - a));
    } while (X + Y > 1.0f);
    E = -logf(curand_uniform(state));
    return (X * E) / (X + Y);
}

// Device function: Best's generator for gamma (a >= 1)
__device__ float gamma_best(float a, curandState_t* state) {
    float b = a - 1.0f;
    float c = 3.0f * a - 0.75f;
    float U, V, W, Y, X, Z;
    do {
        U = curand_uniform(state);
        V = curand_uniform(state);
        W = U * (1.0f - U);
        Y = sqrtf(c / W) * (U - 0.5f);
        X = b + Y;
        if (X < 0.0f) continue;
        Z = 64.0f * powf(W, 3.0f) * powf(V, 3.0f);
    } while (logf(Z) > 2.0f * (b * logf(X / b) - Y));
    return X;
}

// Device function: Gamma sampler (selects Johnk or Best)
__device__ float gamma_sample(float a, curandState_t* state) {
    if (a < 1.0f) return gamma_johnk(a, state);
    else return gamma_best(a, state);
}

// Device function: Simulate VG process on a fixed grid (Algorithm 6.11)
__device__ float simulate_vg(float T, int n, float sigma, float theta, float kappa, curandState_t* state) {
    float dt = T / n;
    float sum_X = 0.0f;
    for (int i = 0; i < n; ++i) {
        float a = dt / kappa;
        float delta_S = kappa * gamma_sample(a, state);
        float N = curand_normal(state);
        float delta_X = sigma * N * sqrtf(delta_S) + theta * delta_S;
        sum_X += delta_X;
    }
    // Martingale adjustment term
    float w = logf(1.0f - theta * kappa - 0.5f * kappa * sigma * sigma) / kappa;
    return w * T + sum_X;
}
