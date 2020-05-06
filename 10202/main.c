#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

#define maxN 18

int N;
char position[maxN][maxN];
void printPlacedCol(int placedCol[]) {
    printf("position now:\n");
    for (int i=0;i<N;i++) {
        for(int j=0; j<N; j++) {
            if(placedCol[i] != j) printf(".");
            else printf("#");
        }
        printf("\n");
    }
    printf("\n");
}
int conflict(int placedCol[], int row, int col) {
    for (int i=0; i<row; i++) {
        if (placedCol[i] == col
            || (row - i == abs(col - placedCol[i]))) return 1;
    }
    return 0;
}

int solve(int placedCol[], int row) {
    // printf("solve(%d)\n", row);
    // printPlacedCol(placedCol);
    if(row == N) { return 1;}
    int ans = 0;
    for (int j=0; j<N; j++) {
        if (position[row][j] == '.' && !conflict(placedCol, row, j)) {
            placedCol[row] = j;
            ans += solve(placedCol, row+1);
        }
    }
    return ans;
}

int placedCol[maxN];
int main() {
    int tc = 1;
    while(scanf("%d", &N) == 1) {
        for (int i=0; i<N; i++) {
            scanf("%s", position[i]);
        }
        int ans = 0;
        #pragma omp parallel for num_threads(1024) collapse(3) private(placedCol) reduction(+ : ans)
        for (int i=0; i<N; i++) {
            for (int j=0; j<N; j++) {
                for (int k=0; k< N; k++) {
                    if (position[0][i] != '.' || position[1][j] != '.' || position[2][k] != '.') { continue; }
                    placedCol[0] = i;
                    if (!conflict(placedCol, 1, j)) {
                        placedCol[1] = j;
                        if (!conflict(placedCol, 2, k)) {
                            placedCol[2] = k;
                            ans += solve(placedCol, 3);
                        }
                    }
                }
            }
        }
        printf("Case %d: %d\n", tc, ans);
        tc++;
    }
}