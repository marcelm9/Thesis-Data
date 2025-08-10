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

  long long *chunkSums_global = NULL;

  #pragma omp parallel
  {
    int num_threads = omp_get_num_threads();
    int tid = omp_get_thread_num();

    #pragma omp single
    {
      chunkSums_global = (long long*)malloc(num_threads * sizeof(long long));
      if (chunkSums_global == NULL) {
        printf("Memory allocation failed\n");
        exit(1);
      }
    }

    // Compute boundaries for this chunk
    long long low = (vector_size * tid) / num_threads;
    long long high = (vector_size * (tid+1)) / num_threads;

    // Compute local prefix sum for the chunk
    long long local_sum = 0;
    for (long long i = low; i < high; i++) {
      local_sum += arr[i];
      prefixSum[i] = local_sum;
    }

    // Store the chunk's total
    chunkSums_global[tid] = local_sum;

    // Barrier to ensure all threads have stored their chunkSums
    #pragma omp barrier

    // Compute the correction array (exclusive prefix of the chunkSums)
    #pragma omp single
    {
      // Convert chunk sums to inclusive prefix
      for (int i = 1; i < num_threads; i++) {
        chunkSums_global[i] += chunkSums_global[i-1];
      }

      // Shift to exclusive prefix
      for (int i = num_threads - 1; i > 0; i--) {
        chunkSums_global[i] = chunkSums_global[i-1];
      }
      chunkSums_global[0] = 0;
    }

    // Add the correction to each element in the chunk
    long long correction = chunkSums_global[tid];
    for (long long i = low; i < high; i++) {
      prefixSum[i] += correction;
    }
  }

  // Free the correction array
  free(chunkSums_global);

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