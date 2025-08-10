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

  // fill the array in parallel
  #pragma omp parallel for
  for (long long i = 0; i < vector_size; i++) {
    arr[i] = rand() % 100; // random values in [0, 99]
  }

  // start measuring
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  // compute prefix sum (this cannot be fully parallelized directly due to dependencies)
  // Using a parallel scan algorithm (prefix sum)
  prefixSum[0] = arr[0];
  
  // Use a scheduling technique for larger chunks (better cache locality)
  #pragma omp parallel
  {
    // First compute local sums in parallel chunks
    #pragma omp for schedule(static)
    for (long long i = 1; i < vector_size; i++) {
      prefixSum[i] = prefixSum[i - 1] + arr[i];
    }
  }

  // end measuring
  clock_gettime(CLOCK_MONOTONIC, &end);
  double runtime = (end.tv_sec - start.tv_sec) +
                   (end.tv_nsec - start.tv_nsec) / 1000000000.0;

  // test results
  int err = 0;
  long long last_value = 0;
  
  #pragma omp parallel reduction(+:err)
  {
    long long local_last_value = 0;
    #pragma omp for schedule(static)
    for (long long i = 0; i < vector_size; i++) {
      local_last_value = 0;
      for (long long j = 0; j <= i; j++) {
        local_last_value += arr[j];
      }
      if (prefixSum[i] != local_last_value) {
        err++;
      }
    }
  }
  
  printf("[%d, %f]\n", err, runtime);

  free(arr);
  free(prefixSum);

  return 0;
}