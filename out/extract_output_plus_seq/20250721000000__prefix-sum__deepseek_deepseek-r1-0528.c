#define COMPILERFLAGS "-lm -fopenmp"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <vector_size> <seed>\n", argv[0]);
    return 1;
  }

  // parse vector size
  char *endptr;
  long long vector_size = strtoll(argv[1], &endptr, 10);
  if (*endptr != '\0' || vector_size <= 0) {
    printf("Invalid vector size\n");
    return 1;
  }

  long long seed = strtoll(argv[2], NULL, 10);
  srand((unsigned int)seed);

  // create arrays
  int *arr = malloc(vector_size * sizeof(int));
  long long *prefixSum = malloc(vector_size * sizeof(long long));
  if (!arr || !prefixSum) {
    printf("Memory allocation failed\n");
    return 1;
  }

  // fill the array
  for (long long i = 0; i < vector_size; i++) {
    arr[i] = rand() % 100;
  }

  // start measuring
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  // compute prefix sum: parallel version
  if (vector_size > 0) {
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int num_threads = omp_get_num_threads();
      long long chunk_size = (vector_size + num_threads - 1) / num_threads;
      long long start_index = tid * chunk_size;
      long long end_index = (tid+1) * chunk_size;
      if (end_index > vector_size) end_index = vector_size;

      long long *chunk_sums = NULL;
      #pragma omp single
      {
          chunk_sums = (long long*)malloc(num_threads * sizeof(long long));
      }

      if (start_index < end_index) {
        prefixSum[start_index] = arr[start_index];
        for (long long i = start_index+1; i < end_index; i++) {
          prefixSum[i] = prefixSum[i-1] + arr[i];
        }
        chunk_sums[tid] = prefixSum[end_index-1];
      } else {
        chunk_sums[tid] = 0;
      }

      #pragma omp barrier
      #pragma omp single
      {
        long long s = 0;
        for (int i = 0; i < num_threads; i++) {
          long long temp = chunk_sums[i];
          chunk_sums[i] = s;
          s += temp;
        }
      }

      if (start_index < end_index) {
        long long correction = chunk_sums[tid];
        for (long long i = start_index; i < end_index; i++) {
          prefixSum[i] += correction;
        }
      }

      #pragma omp barrier
      #pragma omp single
      {
          free(chunk_sums);
      }
    }
  }

  // end measuring
  clock_gettime(CLOCK_MONOTONIC, &end);
  double runtime = (end.tv_sec - start.tv_sec) +
                   (end.tv_nsec - start.tv_nsec) / 1000000000.0;

  // test results
  int err = 0;
  long long last_value = 0;
  for (long long i = 0; i < vector_size; i++) {
    last_value += arr[i];
    if (prefixSum[i] != last_value) {
      err++;
    }
  }
  printf("[%d, %f]\n", err, runtime);

  free(arr);
  free(prefixSum);

  return 0;
}