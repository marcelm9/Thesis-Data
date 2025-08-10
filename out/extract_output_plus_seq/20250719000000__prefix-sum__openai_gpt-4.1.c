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
    unsigned int thread_seed = seed + omp_get_thread_num() * 10007;
    #pragma omp for
    for (long long i = 0; i < vector_size; i++) {
      arr[i] = rand_r(&thread_seed) % 100; // random values in [0, 99]
    }
  }

  // start measuring
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  // parallel prefix sum (Blelloch scan, inclusive)
  int num_threads;
  long long *thread_sums = NULL;

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    #pragma omp single
    {
      num_threads = nthreads;
      thread_sums = malloc(sizeof(long long) * num_threads);
    }

    long long chunk_start = (vector_size * tid) / nthreads;
    long long chunk_end = (vector_size * (tid + 1)) / nthreads;

    if (chunk_start < chunk_end) {
      prefixSum[chunk_start] = arr[chunk_start];
      for (long long i = chunk_start + 1; i < chunk_end; i++) {
        prefixSum[i] = prefixSum[i - 1] + arr[i];
      }
      thread_sums[tid] = prefixSum[chunk_end - 1];
    } else {
      thread_sums[tid] = 0;
    }
    #pragma omp barrier

    long long local_offset = 0;
    for (int i = 0; i < tid; i++) {
      local_offset += thread_sums[i];
    }

    if (chunk_start < chunk_end && local_offset != 0) {
      for (long long i = chunk_start; i < chunk_end; i++) {
        prefixSum[i] += local_offset;
      }
    }
  }
  free(thread_sums);

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