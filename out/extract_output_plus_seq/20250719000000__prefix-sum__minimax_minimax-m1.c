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

  // compute prefix sum in parallel
  long long num_threads = omp_get_max_threads();
  long long chunk_size = (vector_size + num_threads - 1) / num_threads;

  // Compute chunk sums
  long long *chunk_sums = malloc(num_threads * sizeof(long long));
  for (int i = 0; i < num_threads; i++) {
    chunk_sums[i] = 0;
  }

  #pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    long long start_idx = tid * chunk_size;
    long long end_idx = (tid == num_threads - 1) ? vector_size : start_idx + chunk_size;
    long long sum = 0;
    for (long long i = start_idx; i < end_idx; i++) {
      sum += arr[i];
    }
    chunk_sums[tid] = sum;
  }

  // Compute chunk prefix sums
  long long *chunk_prefix_sums = malloc(num_threads * sizeof(long long));
  chunk_prefix_sums[0] = 0;
  for (int i = 1; i < num_threads; i++) {
    chunk_prefix_sums[i] = chunk_prefix_sums[i-1] + chunk_sums[i-1];
  }

  // Compute local prefix sums and add offset
  #pragma omp parallel num_threads(num_threads)
  {
    int tid = omp_get_thread_num();
    long long start_idx = tid * chunk_size;
    long long end_idx = (tid == num_threads - 1) ? vector_size : start_idx + chunk_size;

    if (start_idx < vector_size) {
      // Compute local prefix sum
      prefixSum[start_idx] = arr[start_idx];
      for (long long i = start_idx + 1; i < end_idx; i++) {
        prefixSum[i] = prefixSum[i-1] + arr[i];
      }

      // Add offset
      long long offset = chunk_prefix_sums[tid];
      for (long long i = start_idx; i < end_idx; i++) {
        prefixSum[i] += offset;
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
  free(chunk_sums);
  free(chunk_prefix_sums);

  return 0;
}