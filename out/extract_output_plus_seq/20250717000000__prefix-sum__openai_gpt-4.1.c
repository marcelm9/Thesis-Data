#define COMPILERFLAGS "-fopenmp -lm"

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
  #pragma omp parallel
  {
    unsigned int thread_seed = seed + omp_get_thread_num();
    #pragma omp for
    for (long long i = 0; i < vector_size; i++) {
      arr[i] = rand_r(&thread_seed) % 100; // random values in [0, 99]
    }
  }

  // start measuring
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  // parallel prefix sum (Blelloch scan, work-efficient)
  long long n = vector_size;

  // Copy arr to prefixSum
  #pragma omp parallel for
  for (long long i = 0; i < n; i++) {
    prefixSum[i] = arr[i];
  }

  // Up-sweep phase
  for (long long d = 1; d < n; d <<= 1) {
    #pragma omp parallel for
    for (long long i = d*2 - 1; i < n; i += d*2) {
      prefixSum[i] += prefixSum[i - d];
    }
  }

  // Set last element to 0 for exclusive scan, then shift right.
  long long total = prefixSum[n-1];
  prefixSum[n-1] = 0;

  // Down-sweep phase
  for (long long d = n >> 1; d >= 1; d >>= 1) {
    #pragma omp parallel for
    for (long long i = d*2 - 1; i < n; i += d*2) {
      long long t = prefixSum[i - d];
      prefixSum[i - d] = prefixSum[i];
      prefixSum[i] += t;
    }
  }

  // Make it inclusive scan
  #pragma omp parallel for
  for (long long i = n - 1; i > 0; i--) {
    prefixSum[i] = prefixSum[i] + arr[i];
  }
  prefixSum[0] = arr[0];

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