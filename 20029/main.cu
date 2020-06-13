#include <bits/stdc++.h>
#include <stdint.h>
#include <omp.h>
#define UINT uint32_t
#define MAXN 1024
#define MAXCASE 512
#define MAXGPU 2

#define DEBUG

// __global__
void multiply(int N, UINT A[][MAXN], UINT B[][MAXN], UINT C[][MAXN]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            UINT sum = 0;    // overflow, let it go.
            for (int k = 0; k < N; k++)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
    }
}

// __global__
void add(int N, UINT A[][MAXN], UINT B[][MAXN], UINT C[][MAXN]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            C[i][j] = A[i][j] + B[i][j];
    }
}

void rand_gen(UINT c, int N, UINT A[][MAXN]) {
    UINT x = 2, n = N*N;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            x = (x * x + c + i + j)%n;
            A[i][j] = x;
        }
    }
}

UINT signature(int N, UINT A[][MAXN]) {
    UINT h = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            h = (h + A[i][j]) * 2654435761LU;
    }
    return h;
}

UINT D_ANS[MAXCASE][2];

void solve(int tc, int N, UINT seedA, UINT seedB) {
    static UINT IN[2][MAXN][MAXN], TMP[6][MAXN][MAXN];
    #pragma omp parallel
    {
        rand_gen(seedA, N, IN[0]);
        rand_gen(seedB, N, IN[1]);
    }
    #pragma omp parallel
    {
        // AB
        multiply(N, IN[0], IN[1], TMP[0]);
        // BA
        multiply(N, IN[1], IN[0], TMP[1]);
        // AB+BA
        add(N, TMP[0], TMP[1], TMP[2]);
        
        // ABA
        multiply(N, TMP[0], IN[0], TMP[3]);
        // BAB
        multiply(N, TMP[1], IN[1], TMP[4]);
        // ABA+BAB
        add(N, TMP[3], TMP[4], TMP[5]);
    }

    D_ANS[tc][0] = signature(N, TMP[2]);
    D_ANS[tc][1] = signature(N, TMP[5]);
 
    return 0;
}

int N[MAXCASE];
UINT seedA[MAXCASE], seedB[MAXCASE];

int main() {
    int tc = 0;
    while (scanf("%d %u %u", &N[tc], &seedA[tc], &seedB[tc]) == 3) {
        tc++;
    }

	omp_set_num_threads(MAXGPU);
#pragma omp parallel for schedule(dynamic)
    for (int i=0; i<tc; i++) {
        cudaSetDevice(omp_get_thread_num());
        solve(i, N[i], seedA[i], seedB[i]);
    }

    for (int i=0; i<tc; i++) {
        printf("%u\n", D_ANS[i][0], D_ANS[i][1]);
    }
    return 0;
}