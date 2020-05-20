#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

#define maxN 2050
# define nLiveNeighbor(A,i,j) \
A [ i + 1][ j ] + A [ i - 1][ j ] + A [ i ][ j + 1] + \
A [ i ][ j - 1] + A [ i + 1][ j + 1] + A [ i + 1][ j - 1] + \
A [ i - 1][ j + 1] + A [ i - 1][ j - 1]

char stateA[maxN][maxN], stateB[maxN][maxN];

int N;

void printState(char state[maxN][maxN]) {
    for (int i=1; i<=N; i++) {
        for (int j=1; j<=N; j++) {
            state[i][j] += '0';
        }
        state[i][N+1] = '\0';
        printf("%s\n", &state[i][1]);
    }
}

inline int updateState(char A[maxN][maxN], char B[maxN][maxN], int x, int y) {
    int number = nLiveNeighbor(A, x, y);
    B[x][y] = (A[x][y] == 0 && number == 3) || (A[x][y] == 1 && (number == 2 || number == 3));
}

void simulate(int rounds) {
    #pragma omp parallel
    for (int round = 1; round <= rounds; round++) {
        int cur = round % 2;
        #pragma omp for
        for (int i=1; i<=N; i++) {
            for (int j=1; j<=N; j++) {
                if (cur) updateState(stateA, stateB, i, j);
                else updateState(stateB, stateA, i, j);
            }
        }
    }
}

int main() {
    int M;
    while(scanf("%d%d", &N, &M) == 2) {
        for (int i=1; i<=N; i++) {
            scanf("%s", &stateA[i][1]);
            stateA[i][0] = 0;
            for (int j=1; j<=N; j++) stateA[i][j] -= '0';
            stateA[i][N+1] = 0;
        }
        simulate(M);
        printState(M % 2 ? stateB : stateA);
    }
}