#include <bits/stdc++.h>
#include <stdint.h>
#include <omp.h>
#define UINT uint32_t
#define MAXN 1024
#define MAXCASE 512

__global__
void multiply(int N, UINT* A, UINT* B, UINT* C) {
    // int row = blockIdx.x;
    // int col = threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    // printf("row=%d col=%d\n",row, col);
    if (row < N && col < N) {
        UINT sum = 0;    // overflow, let it go.
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

__global__
void add(int N, UINT* A, UINT* B, UINT* C) {
    // int row = blockIdx.x;
    // int col = threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        C[row * N + col] = A[row * N + col] + B[row * N + col];
    }
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

UINT signature(int N, UINT* A) {
    UINT h = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            h = (h + A[i * N + j]) * 2654435761LU;
    }
    return h;
}

UINT ANS[MAXCASE][2];

void solve(int tc, int N, UINT seedA, UINT seedB) {
    UINT *D_IN[2], *D_TMP[6];

    for (int i=0; i<2; i++) {
        cudaMalloc(&D_IN[i], N*N*sizeof(UINT));
    }
    for (int i=0; i<6; i++) {
        cudaMalloc(&D_TMP[i], N*N*sizeof(UINT));
    }

    rand_gen<<<1, 1>>>(seedA, N, D_IN[0]);
    rand_gen<<<1, 1>>>(seedB, N, D_IN[1]);

    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);
    if (N*N > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(N)/double(threadsPerBlock.y));
    }

    cudaDeviceSynchronize();
    // AB
    multiply<<<blocksPerGrid, threadsPerBlock>>>(N, D_IN[0], D_IN[1], D_TMP[0]);
    // BA
    multiply<<<blocksPerGrid, threadsPerBlock>>>(N, D_IN[1], D_IN[0], D_TMP[1]);

    cudaDeviceSynchronize();
    // AB+BA
    add<<<blocksPerGrid, threadsPerBlock>>>(N, D_TMP[0], D_TMP[1], D_TMP[2]);
    // ABA
    multiply<<<blocksPerGrid, threadsPerBlock>>>(N, D_TMP[0], D_IN[0], D_TMP[3]);
    // BAB
    multiply<<<blocksPerGrid, threadsPerBlock>>>(N, D_TMP[1], D_IN[1], D_TMP[4]);

    cudaDeviceSynchronize();
    // ABA+BAB
    add<<<blocksPerGrid, threadsPerBlock>>>(N, D_TMP[3], D_TMP[4], D_TMP[5]);


    cudaDeviceSynchronize();

    ANS[tc][0] = signature(N, D_TMP[2]);
    ANS[tc][1] = signature(N, D_TMP[5]);
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