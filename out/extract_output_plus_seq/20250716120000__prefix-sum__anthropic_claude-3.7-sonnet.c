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
  #pragma omp parallel for
  for (long long i = 0; i < vector_size; i++) {
    // Use a thread-safe random number generation approach
    int my_rand;
    #pragma omp critical
    {
      my_rand = rand() % 100; // random values in [0, 99]
    }
    arr[i] = my_rand;
  }

  // start measuring
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  // compute prefix sum (this is a sequential dependency, can't be fully parallelized)
  prefixSum[0] = arr[0];
  for (long long i = 1; i < vector_size; i++) {
    prefixSum[i] = prefixSum[i - 1] + arr[i];
  }

  // end measuring
  clock_gettime(CLOCK_MONOTONIC, &end);
  double runtime = (end.tv_sec - start.tv_sec) +
                   (end.tv_nsec - start.tv_nsec) / 1000000000.0;

  // test results
  int err = 0;
  long long last_value = 0;
  #pragma omp parallel
  {
    #pragma omp for reduction(+:err) private(last_value)
    for (long long i = 0; i < vector_size; i++) {
      // Recalculate prefix sum for checking
      if (i == 0) {
        last_value = arr[0];
      } else {
        last_value = 0;
        for (long long j = 0; j <= i; j++) {
          last_value += arr[j];
        }
      }
      
      if (prefixSum[i] != last_value) {
        err++;
      }
    }
  }
  
  printf("[%d, %f]\n", err, runtime);

  free(arr);
  free(prefixSum);

  return 0;
}