#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <pthread.h>
#include <assert.h>
#include "utils.h"

#define MAXN 10000005
#define MAX_THREAD 6

int n;
uint32_t key;
uint32_t prefix_sum[MAXN];

void print_prefix() {
  printf("original:");
  for (int i=1 ; i<=n; i++) {
    printf("%" PRIu32 " ", encrypt(i, key));
  }
  printf("\n");
  printf("prefix now:");
  for(int i=1; i<=n ; i++) {
    printf("%" PRIu32 " ", prefix_sum[i]);
  }
  printf("\n");
}

void presum_seq(void* thread_num) {
  int num = *((int *)thread_num);
  int L = (n / MAX_THREAD) * num + 1;
  int R = num == MAX_THREAD - 1 ? n : ((n/MAX_THREAD) * (num + 1));
  uint32_t sum = 0;
  // printf("L=%d, R=%d\n", L, R);
  for (int i = L; i <= R; i++) {
    sum += encrypt(i, key);
    prefix_sum[i] = sum;
  }
}

struct thread_data {
  int thread_num;
  uint32_t num_to_add;
};

void addNum(void* data) {
  int thread_num =  ((struct thread_data*) data) -> thread_num;
  uint32_t num_to_add = ((struct thread_data*) data) -> num_to_add;
  int L = (n / MAX_THREAD) * thread_num + 1;
  int R = thread_num == MAX_THREAD - 1 ? n : ((n/MAX_THREAD) * (thread_num + 1));
  for (int i=L ; i<R; i++) {
    prefix_sum[i] += num_to_add;
  }
}

int main() {
  // Init
  pthread_t threads[MAX_THREAD];
  int* thread_num = (int *)calloc(MAX_THREAD, sizeof(int));
  for(int i=0; i<MAX_THREAD; i++) {
    thread_num[i] = i;
  }
  struct thread_data* thread_array = malloc(MAX_THREAD * sizeof(struct thread_data));
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  while (scanf("%d %" PRIu32, &n, &key) == 2) {
    // First stage: Prefix sum for each thread
    for (int i = 0; i < MAX_THREAD; i++) {
      int error = pthread_create(&threads[i], &attr, presum_seq, (void *) (&thread_num[i]));
      assert(error == 0);
    }
    for (int i=0; i<MAX_THREAD; i++) pthread_join(threads[i], NULL);
    // print_prefix();

    // Second stage: Parallel prefix sum algoithm
    // But I make it sequential here since the operation number is quiet small...
    
    // struct thread_data* thread_array = malloc(MAX_THREAD * sizeof(struct thread_data));
    // for (int stage=1; stage<MAX_THREAD; stage <<= 1) {
    //   for (int i=0; i<MAX_THREAD; i++) {
    //     int error= pthread_create(&threads[i], &attr, presum_parallel, (void *) &thread_array[i]);
    //     assert(error == 0);
    //   }
    //   for (int i=0; i<MAX_THREAD; i++) pthread_join(threads[i], NULL);
    // }
    uint32_t sum = 0;
    for (int i=0; i<MAX_THREAD; i++) {
      int R = i == MAX_THREAD - 1 ? n : ((n/MAX_THREAD) * (i + 1));
      int num = prefix_sum[R];
      prefix_sum[R] += sum;
      sum += num;
    }
    // print_prefix();

    // Third stage: Adjust every part
    for (int i = 1; i < MAX_THREAD; i++) {
      thread_array[i].num_to_add = prefix_sum[(n / MAX_THREAD) * i];
      thread_array[i].thread_num = i;
    }
    for (int i = 1; i < MAX_THREAD; i++) {
      int error = pthread_create(&threads[i], &attr, addNum, (void *) &thread_array[i]);
      assert(error == 0);
    }
    for (int i=1; i<MAX_THREAD; i++) pthread_join(threads[i], NULL);
    // print_prefix();

    output(prefix_sum, n);
  }
  free(thread_array);
  free(thread_num);
  pthread_exit(NULL);
  return 0;
}
