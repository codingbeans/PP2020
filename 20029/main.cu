#include <bits/stdc++.h>
#include <stdint.h>
#include <omp.h>
#define UINT uint32_t
#define MAXN 1024
#define MAXCASE 512

#define DEBUG

__global__
void multiply(int N, UINT* A, UINT* B, UINT* C) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    if (row < N && col < N) {
            UINT sum = 0;    // overflow, let it go.
            for (int k = 0; k < N; k++)
                sum += A[row * N + k] * B[k * N + col];
            C[row * N + col] = sum;
    }
    //     }
    // }
}

__global__
void add(int N, UINT* A, UINT* B, UINT* C) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++)
    if (row < N && col < N) {
            C[row * N + col] = A[row * N + col] + B[row * N + col];
    }
    // }
}

__global__
void rand_gen(UINT c, int N, UINT* A) {
    UINT x = 2, n = N*N;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            x = (x * x + c + i + j)%n;
            A[i * N + j] = x;
        }
    }
}

__global__
void signature(int N, UINT* A, UINT* ans) {
    UINT h = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            h = (h + A[i * N + j]) * 2654435761LU;
    }
    *ans = h;
}

UINT ANS[MAXCASE][2];

void solve(int tc, int N, UINT seedA, UINT seedB) {
    // UINT IN[2][MAXN][MAXN], TMP[6][MAXN][MAXN];
    // UINT *IN, *TMP;
    UINT *D_IN[2], *D_TMP[6], *D_ANS;
    // IN = (UINT*)malloc(2*MAXN*MAXN*sizeof(UINT));
    // TMP = (UINT*)malloc(6*MAXN*MAXN*sizeof(UINT));

    cudaMalloc(&D_ANS, 2*sizeof(UINT));
    #pragma omp parallel for 
    for (int i=0; i<2; i++) {
        cudaMalloc(&D_IN[i], N*N*sizeof(UINT));
    }

    #pragma omp parallel for 
    for (int i=0; i<6; i++) {
        cudaMalloc(&D_TMP[i], N*N*sizeof(UINT));
    }
    // #pragma omp parallel
    // {
        // rand_gen(seedA, N, IN[0]);
        // rand_gen(seedB, N, IN[1]);
        rand_gen<<<1, 1>>>(seedA, N, D_IN[0]);
        rand_gen<<<1, 1>>>(seedB, N, D_IN[1]);
    // }


    cudaDeviceSynchronize();
    // AB
    // multiply(N, IN[0], IN[1], TMP[0]);
    multiply<<<N, N>>>(N, D_IN[0], D_IN[1], D_TMP[0]);
    // BA
    // multiply(N, IN[1], IN[0], TMP[1]);
    multiply<<<N, N>>>(N, D_IN[1], D_IN[0], D_TMP[1]);

    cudaDeviceSynchronize();
    // AB+BA
    // add(N, TMP[0], TMP[1], TMP[2]);
    add<<<N, N>>>(N, D_TMP[0], D_TMP[1], D_TMP[2]);
    

    // ABA
    // multiply(N, TMP[0], IN[0], TMP[3]);
    multiply<<<N, N>>>(N, D_TMP[0], D_IN[0], D_TMP[3]);
    // BAB
    // multiply(N, TMP[1], IN[1], TMP[4]);
    multiply<<<N, N>>>(N, D_TMP[1], D_IN[1], D_TMP[4]);

    cudaDeviceSynchronize();
    // ABA+BAB
    // add(N, TMP[3], TMP[4], TMP[5]);
    add<<<N, N>>>(N, D_TMP[3], D_TMP[4], D_TMP[5]);


    cudaDeviceSynchronize();
    // #pragma omp parallel
    // {
        // D_ANS[tc][0] = signature(N, TMP[2]);
        // D_ANS[tc][1] = signature(N, TMP[5]);
        signature<<<1, 1>>>(N, D_TMP[2], &D_ANS[0]);
        signature<<<1, 1>>>(N, D_TMP[5], &D_ANS[1]);
    // }
 
    cudaMemcpy(ANS[tc], D_ANS, 2 * sizeof(UINT), cudaMemcpyDeviceToHost);
    return;
}

int N[MAXCASE];
UINT seedA[MAXCASE], seedB[MAXCASE];

int main() {
    int tc = 0;
    while (scanf("%d %u %u", &N[tc], &seedA[tc], &seedB[tc]) == 3) {
        tc++;
    }

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
	omp_set_num_threads(deviceCount);
#pragma omp parallel for schedule(dynamic)
    for (int i=0; i<tc; i++) {
        cudaSetDevice(omp_get_thread_num());
        solve(i, N[i], seedA[i], seedB[i]);
    }

    for (int i=0; i<tc; i++) {
        printf("%u\n%u\n", ANS[i][0], ANS[i][1]);
    }
    return 0;
}