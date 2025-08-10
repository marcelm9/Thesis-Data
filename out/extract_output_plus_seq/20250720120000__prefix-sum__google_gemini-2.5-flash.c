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

  // compute prefix sum
  // This is a classic prefix sum (scan) problem.
  // A simple parallelization of the sequential loop results in a race condition.
  // A common parallel approach is a two-pass algorithm or a tree-based sum.
  // For simplicity and correctness with OpenMP, we'll use a two-pass algorithm.

  // Pass 1: Compute partial sums in blocks
  int num_threads = omp_get_max_threads();
  if (vector_size < num_threads) {
      num_threads = vector_size;
  }
  long long *block_sums = (long long *)calloc(num_threads, sizeof(long long)); // Initialize to 0

  if (!block_sums) {
      printf("Memory allocation failed for block_sums\n");
      free(arr);
      free(prefixSum);
      return 1;
  }

  #pragma omp parallel num_threads(num_threads)
  {
      int tid = omp_get_thread_num();
      long long local_sum = 0;
      long long start_idx = tid * (vector_size / num_threads);
      long long end_idx = (tid == num_threads - 1) ? vector_size : (tid + 1) * (vector_size / num_threads);

      // Compute local prefix sums and the total sum for this block
      if (start_idx < end_idx) { // Ensure there's work to do
          prefixSum[start_idx] = arr[start_idx];
          local_sum = arr[start_idx];
          for (long long i = start_idx + 1; i < end_idx; i++) {
              prefixSum[i] = prefixSum[i - 1] + arr[i];
              local_sum += arr[i];
          }
      }
      block_sums[tid] = local_sum; // Store the total sum of this block
  }

  // Pass 2: Adjust sums based on previous block sums sequentially
  // This part is inherently sequential due to dependencies.
  // However, it's a small overhead compared to the first pass for large N.
  for (int i = 1; i < num_threads; i++) {
      block_sums[i] += block_sums[i-1];
  }

  #pragma omp parallel num_threads(num_threads)
  {
      int tid = omp_get_thread_num();
      long long start_idx = tid * (vector_size / num_threads);
      long long end_idx = (tid == num_threads - 1) ? vector_size : (tid + 1) * (vector_size / num_threads);

      long long offset = 0;
      if (tid > 0) {
          offset = block_sums[tid - 1]; // Offset from the previous block's total sum
      }

      if (start_idx < end_idx) { // Ensure there's work to do
          for (long long i = start_idx; i < end_idx; i++) {
              prefixSum[i] += offset;
          }
      }
  }

  free(block_sums);


  // end measuring
  clock_gettime(CLOCK_MONOTONIC, &end);
  double runtime = (end.tv_sec - start.tv_tv_sec) +
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