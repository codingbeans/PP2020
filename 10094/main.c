#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

#define maxN 10000
#define maxSize 1000000 + 5
int dp[maxSize][2];

int main() {
    int N, M;
    scanf("%d%d", &N, &M);
    int curRow = 0;
    for (int i=0; i<N; i++) {
        int W, V;
        scanf("%d%d", &W, &V);
        #pragma omp parallel for
        for (int i=0; i<=M; i++) {
            if (dp[i][curRow ^ 1] < dp[i][curRow]) {
                dp[i][curRow ^ 1] = dp[i][curRow];
            }
            if (i+W <= M && (dp[i+W][curRow ^ 1] < dp[i][curRow] + V)) {
                dp[i+W][curRow ^ 1] = dp[i][curRow] + V;
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
        if (ans < dp[i][curRow]) {
            ans = dp[i][curRow];
        }
    }
    printf("%d\n", ans);
}