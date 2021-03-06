#include <bits/stdc++.h>
#include <stdint.h>
#include <omp.h>
#define UINT uint32_t
#define MAXN 1024
#define MAXCASE 512
#define BLOCKSIZE 16

__global__
void multiply(int N, UINT* A, UINT* B, UINT* C) {
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    if (row < N && col < N) {
        UINT sum = 0;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

__global__
void add(int N, UINT* A, UINT* B, UINT* C) {
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    if (row < N && col < N) {
        C[row * N + col] = A[row * N + col] + B[row * N + col];
    }
}

// __global__
void rand_gen(UINT c, int N, UINT* A) {
    UINT x = 2, n = N*N;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            x = (x * x + c + i + j)%n;
            A[i * N + j] = x;
        }
    }
}

// __global__
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
    UINT *IN[2], *TMP[2];
    UINT *D_IN[2], *D_TMP[6], *D_ANS;

    cudaMalloc(&D_ANS, 2*sizeof(UINT));
    for (int i=0; i<2; i++) {
        IN[i] = (UINT*)malloc(N*N*sizeof(UINT));
        cudaMalloc(&D_IN[i], N*N*sizeof(UINT));
    }
    for (int i=0; i<6; i++) {
        cudaMalloc(&D_TMP[i], N*N*sizeof(UINT));
    }
    rand_gen(seedA, N, IN[0]);
    rand_gen(seedB, N, IN[1]);
    cudaMemcpy(D_IN[0], IN[0], N*N*sizeof(UINT), cudaMemcpyHostToDevice);
    cudaMemcpy(D_IN[1], IN[1], N*N*sizeof(UINT), cudaMemcpyHostToDevice);
    for (int i=0; i<2; i++) {
        free(IN[i]);
    }
    // rand_gen<<<1, 1>>>(seedA, N, D_IN[0]);
    // rand_gen<<<1, 1>>>(seedB, N, D_IN[1]);

    dim3 threadsPerBlock(BLOCKSIZE, BLOCKSIZE);
    dim3 blocksPerGrid((N + BLOCKSIZE - 1) / BLOCKSIZE, (N + BLOCKSIZE -1) / BLOCKSIZE);

    // AB
    multiply<<<blocksPerGrid, threadsPerBlock>>>(N, D_IN[0], D_IN[1], D_TMP[0]);
    // BA
    multiply<<<blocksPerGrid, threadsPerBlock>>>(N, D_IN[1], D_IN[0], D_TMP[1]);

    // AB+BA
    add<<<blocksPerGrid, threadsPerBlock>>>(N, D_TMP[0], D_TMP[1], D_TMP[2]);

    // ABA
    multiply<<<blocksPerGrid, threadsPerBlock>>>(N, D_TMP[0], D_IN[0], D_TMP[3]);
    // BAB
    multiply<<<blocksPerGrid, threadsPerBlock>>>(N, D_TMP[1], D_IN[1], D_TMP[4]);

    // ABA+BAB
    add<<<blocksPerGrid, threadsPerBlock>>>(N, D_TMP[3], D_TMP[4], D_TMP[5]);

    for (int i=0; i<2; i++) {
        TMP[i] = (UINT*)malloc(N*N*sizeof(UINT));
    }
    cudaMemcpy(TMP[0], D_TMP[2], N*N*sizeof(UINT), cudaMemcpyDeviceToHost);
    cudaMemcpy(TMP[1], D_TMP[5], N*N*sizeof(UINT), cudaMemcpyDeviceToHost);
    signature(N, TMP[0], &ANS[tc][0]);
    signature(N, TMP[1], &ANS[tc][1]);
    // signature<<<1, 1>>>(N, D_TMP[2], &D_ANS[0]);
    // signature<<<1, 1>>>(N, D_TMP[5], &D_ANS[1]);
 
    // cudaMemcpy(ANS[tc], D_ANS, 2 * sizeof(UINT), cudaMemcpyDeviceToHost);

    cudaFree(D_ANS);
    for (int i=0; i<2; i++) {
        free(TMP[i]);
        cudaFree(D_IN[i]);
    }
    for (int i=0; i<6; i++) {
        cudaFree(D_TMP[i]);
    }
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
        cudaSetDevice(omp_get_thread_num() % deviceCount);
        solve(i, N[i], seedA[i], seedB[i]);
    }

    for (int i=0; i<tc; i++) {
        printf("%u\n%u\n", ANS[i][0], ANS[i][1]);
    }
    return 0;
}