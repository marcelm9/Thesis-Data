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

  // compute prefix sum using OpenMP parallel scan (prefix sum) with a parallel for loop
  
  // This is a common and efficient way to parallelize a prefix sum (scan) operation.
  // It uses a two-pass approach.
  
  // Pass 1: Local prefix sums within chunks
  int num_threads = omp_get_max_threads();
  if (vector_size < num_threads) { // Handle small vector sizes
    num_threads = 1;
  }
  long long *block_sums = calloc(num_threads, sizeof(long long)); // Stores the sum of each thread's block

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads(); // Actual number of threads used
    
    long long chunk_size = (vector_size + nthreads - 1) / nthreads;
    long long start_idx = tid * chunk_size;
    long long end_idx = (tid + 1) * chunk_size;
    if (end_idx > vector_size) {
      end_idx = vector_size;
    }

    if (start_idx < end_idx) {
      prefixSum[start_idx] = arr[start_idx];
      for (long long i = start_idx + 1; i < end_idx; i++) {
        prefixSum[i] = prefixSum[i - 1] + arr[i];
      }
      block_sums[tid] = prefixSum[end_idx - 1]; // Store the last element of the local prefix sum
    } else {
      block_sums[tid] = 0; // If chunk is empty
    }
  }

  // Pass 2: Accumulate global sums
  // Compute prefix sum of block_sums
  for (int i = 1; i < num_threads; i++) {
    block_sums[i] += block_sums[i - 1];
  }

  // Add accumulated sums to subsequent blocks
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    long long chunk_size = (vector_size + nthreads - 1) / nthreads;
    long long start_idx = tid * chunk_size;
    long long end_idx = (tid + 1) * chunk_size;
    if (end_idx > vector_size) {
      end_idx = vector_size;
    }

    long long offset = (tid > 0) ? block_sums[tid - 1] : 0;

    if (start_idx < end_idx) {
        for (long long i = start_idx; i < end_idx; i++) {
            prefixSum[i] += offset;
        }
    }
  }

  free(block_sums);

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