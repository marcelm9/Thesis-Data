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

  // compute prefix sum in parallel using OpenMP
  int num_threads = omp_get_max_threads();
  long long num_blocks = num_threads;
  long long chunk_size = (vector_size + num_blocks - 1) / num_blocks;

  long long *block_sums = malloc(num_blocks * sizeof(long long));
  if (!block_sums) {
    printf("Memory allocation failed for block_sums\n");
    free(arr);
    free(prefixSum);
    return 1;
  }

  #pragma omp parallel for
  for (long long b = 0; b < num_blocks; b++) {
    long long start = b * chunk_size;
    long long end = (b == num_blocks - 1) ? vector_size : start + chunk_size;
    long long sum = 0;
    for (long long i = start; i < end; i++) {
      sum += arr[i];
    }
    block_sums[b] = sum;
  }

  long long *block_prefix_sums = malloc(num_blocks * sizeof(long long));
  if (!block_prefix_sums) {
    printf("Memory allocation failed for block_prefix_sums\n");
    free(arr);
    free(prefixSum);
    free(block_sums);
    return 1;
  }

  if (num_blocks > 0) {
    block_prefix_sums[0] = block_sums[0];
    for (long long b = 1; b < num_blocks; b++) {
      block_prefix_sums[b] = block_prefix_sums[b-1] + block_sums[b];
    }
  }

  #pragma omp parallel for
  for (long long b = 0; b < num_blocks; b++) {
    long long start = b * chunk_size;
    long long end = (b == num_blocks - 1) ? vector_size : start + chunk_size;
    long long offset = (b == 0) ? 0 : block_prefix_sums[b-1];
    long long current_sum = offset;
    for (long long i = start; i < end; i++) {
      current_sum += arr[i];
      prefixSum[i] = current_sum;
    }
  }

  free(block_sums);
  free(block_prefix_sums);

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