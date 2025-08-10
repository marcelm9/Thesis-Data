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

  // create arrays
  int *arr = malloc(vector_size * sizeof(int));
  long long *prefixSum = malloc(vector_size * sizeof(long long));
  if (!arr || !prefixSum) {
    printf("Memory allocation failed\n");
    return 1;
  }

  // fill the array in parallel
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int num_threads = omp_get_num_threads();
    long long chunk_size = vector_size / num_threads;
    long long start = tid * chunk_size;
    long long end = (tid == num_threads - 1) ? vector_size : start + chunk_size;

    // compute starting seed for this chunk
    unsigned int seed_chunk = (unsigned int)seed;
    for (long long i = 0; i < start; i++) {
      seed_chunk = (seed_chunk * 1103515245 + 12345) % (1LL << 31);
    }

    for (long long i = start; i < end; i++) {
      seed_chunk = (seed_chunk * 1103515245 + 12345) % (1LL << 31);
      arr[i] = seed_chunk % 100;
    }
  }

  // start measuring
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  // compute prefix sum
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