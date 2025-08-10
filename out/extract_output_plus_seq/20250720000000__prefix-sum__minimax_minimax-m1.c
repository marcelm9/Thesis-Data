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
    arr[i] = rand() % 100; // random values in [0, 99]
  }

  // start measuring
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  // compute prefix sum using OpenMP
  int num_threads = omp_get_max_threads();
  long long chunk_size = vector_size / num_threads;
  if (vector_size % num_threads != 0) {
    chunk_size++;
  }

  long long *chunk_sums = malloc(num_threads * sizeof(long long));
  #pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    long long start_idx = tid * chunk_size;
    long long end_idx = (tid == num_threads - 1) ? vector_size : (tid + 1) * chunk_size;
    long long sum = 0;
    for (long long i = start_idx; i < end_idx; i++) {
      sum += arr[i];
    }
    chunk_sums[tid] = sum;
  }

  // compute chunk prefix sums
  long long *chunk_prefix_sums = malloc(num_threads * sizeof(long long));
  chunk_prefix_sums[0] = chunk_sums[0];
  for (int i = 1; i < num_threads; i++) {
    chunk_prefix_sums[i] = chunk_prefix_sums[i-1] + chunk_sums[i];
  }

  // compute prefix sums per chunk
  #pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    long long start_idx = tid * chunk_size;
    long long end_idx = (tid == num_threads - 1) ? vector_size : (tid + 1) * chunk_size;
    long long starting_offset = (tid == 0) ? 0 : chunk_prefix_sums[tid-1];
    long long current_sum = starting_offset;
    for (long long i = start_idx; i < end_idx; i++) {
      current_sum += arr[i];
      prefixSum[i] = current_sum;
    }
  }

  free(chunk_sums);
  free(chunk_prefix_sums);

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