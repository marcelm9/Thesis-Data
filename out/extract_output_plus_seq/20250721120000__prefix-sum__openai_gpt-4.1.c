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
  for (long long i = 0; i < vector_size; i++) {
    arr[i] = rand() % 100; // random values in [0, 99]
  }

  // start measuring
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  // compute prefix sum (parallel version)
  int num_threads;
  #pragma omp parallel
  {
    #pragma omp single
    num_threads = omp_get_num_threads();
  }

  long long *block_sum = malloc(num_threads * sizeof(long long));
  if (!block_sum) {
    printf("Memory allocation failed\n");
    free(arr);
    free(prefixSum);
    return 1;
  }

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    long long chunk_start, chunk_end;
    long long chunk_size = vector_size / num_threads;
    chunk_start = tid * chunk_size;
    chunk_end = (tid == num_threads - 1) ? vector_size : (chunk_start + chunk_size);

    // Step 1: Every thread computes a local prefix sum in its own chunk
    if (chunk_start < chunk_end) {
      prefixSum[chunk_start] = arr[chunk_start];
      for (long long i = chunk_start + 1; i < chunk_end; i++) {
        prefixSum[i] = prefixSum[i - 1] + arr[i];
      }
      block_sum[tid] = prefixSum[chunk_end - 1];
    } else {
      block_sum[tid] = 0;
    }

    #pragma omp barrier

    // Step 2: Compute prefix sum of the block sums (serial, small)
    #pragma omp single
    {
      for (int i = 1; i < num_threads; i++) {
        block_sum[i] += block_sum[i - 1];
      }
    }
    #pragma omp barrier

    // Step 3: Add offset to each chunk (except first chunk)
    if (tid > 0 && chunk_start < chunk_end) {
      long long offset = block_sum[tid - 1];
      for (long long i = chunk_start; i < chunk_end; i++) {
        prefixSum[i] += offset;
      }
    }
  }

  free(block_sum);

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