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
  #pragma omp parallel for schedule(static)
  for (long long i = 0; i < vector_size; i++) {
    arr[i] = rand() % 100; // random values in [0, 99]
  }

  // start measuring
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  // OpenMP parallel prefix sum (scan) using Blelloch algorithm (up-sweep and down-sweep)
  // Only applies when vector_size > 1
  if (vector_size > 0) {
    for (long long i = 0; i < vector_size; i++)
      prefixSum[i] = arr[i];
    long long d, twod;
    // Up-sweep (reduce) phase
    for (d = 1; d < vector_size; d <<= 1) {
      #pragma omp parallel for schedule(static)
      for (long long i = d * 2 - 1; i < vector_size; i += d * 2) {
        prefixSum[i] += prefixSum[i - d];
      }
    }
    // Down-sweep phase
    long long last = prefixSum[vector_size - 1];
    prefixSum[vector_size - 1] = arr[vector_size - 1];
    for (d = vector_size / 2; d >= 1; d >>= 1) {
      #pragma omp parallel for schedule(static)
      for (long long i = d * 2 - 1; i < vector_size; i += d * 2) {
        long long t = prefixSum[i - d];
        prefixSum[i - d] = prefixSum[i];
        prefixSum[i] = prefixSum[i] + t;
      }
    }
    // Now prefixSum[i] = arr[0]+...+arr[i]
    prefixSum[0] = arr[0];
    for (long long i = 1; i < vector_size; i++)
      prefixSum[i] += arr[0] - arr[i]; // fix for downsweep root value
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