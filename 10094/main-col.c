#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

#define maxN 10000
#define maxSize 1000000 + 5
int dp[2][maxSize];

int main() {
    int N, M;
    scanf("%d%d", &N, &M);
    int curRow = 0;
    for (int i=0; i<N; i++) {
        int W, V;
        scanf("%d%d", &W, &V);
        #pragma omp parallel for
        for (int i=0; i<=M; i++) {
            if (dp[curRow ^ 1][i] < dp[curRow][i]) {
                dp[curRow ^ 1][i] = dp[curRow][i];
            }
            if (i+W <= M && (dp[curRow ^ 1][i+W] < dp[curRow][i] + V)) {
                dp[curRow ^ 1][i+W] = dp[curRow][i] + V;
            }
        }
        // for (int i=0; i<=M; i++) {
        //     printf("%d%c", dp[i][curRow], i==M ? '\n': ' ');
        // }
        // for (int i=0; i<=M; i++) {
        //     printf("%d%c", dp[i][curRow ^ 1], i==M ? '\n': ' ');
        // }
        curRow ^= 1;
    }
    int ans = 0;
    for (int i=0; i<=M; i++) {
        if (ans < dp[curRow][i]) {
            ans = dp[curRow][i];
        }
    }
    printf("%d\n", ans);
}