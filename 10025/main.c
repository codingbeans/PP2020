#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

#define maxSize 500

int A_h, A_w, B_h, B_w;
int A[maxSize][maxSize], B[maxSize][maxSize];

long long cal(int x, int y) {
    long long sum = 0;
#pragma omp parallel for reduction (+ : sum)
    for (int i=0; i<B_h; i++)
        for (int j=0; j<B_w; j++) {
            sum += (A[x+i][y+j] - B[i][j]) * (A[x+i][y+j] - B[i][j]);
        }
    return sum;
}

long long best[maxSize];
int best_y[maxSize];

void solve() {
#pragma omp parallel for
    for (int i=0; i<=A_h - B_h; i++)
    {
        best[i] = ((long long) 1) << 60;
        for (int j=0; j<=A_w - B_w; j++) {
            long long now = cal(i, j);
            // printf("i=%d, j=%d, now=%d, best=%d\n", i, j, now, best);
            if (now < best[i])
            {
                best[i] = now;
                best_y[i] = j;
            }
        }
    }
    int glob_best_x = 0;
    for (int i=1; i<= A_h - B_h; i++) {
        if (best[i] < best[glob_best_x]) {
            glob_best_x = i;
        }
    }
    printf("%d %d\n", glob_best_x + 1, best_y[glob_best_x] + 1);
}

int main() {
    while(scanf("%d%d%d%d", &A_h, &A_w, &B_h, &B_w) == 4) {
        for (int i=0; i<A_h; i++)
            for (int j=0; j<A_w; j++) {
                scanf("%d", &A[i][j]);
            }
        for (int i=0; i<B_h; i++)
            for (int j=0; j<B_w; j++) {
                scanf("%d", &B[i][j]);
            }
        solve();
    }
}