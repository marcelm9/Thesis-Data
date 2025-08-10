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

  // compute prefix sum in parallel using OpenMP
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    long long *chunk_sums = NULL;

    #pragma omp single
    {
        chunk_sums = (long long *)malloc(nthreads * sizeof(long long));
    }

    long long chunk_start = (vector_size * tid) / nthreads;
    long long chunk_end = (vector_size * (tid + 1)) / nthreads;
    long long chunk_size = chunk_end - chunk_start;

    if (chunk_size > 0) {
        prefixSum[chunk_start] = arr[chunk_start];
        for (long long i = chunk_start + 1; i < chunk_end; i++) {
            prefixSum[i] = prefixSum[i - 1] + arr[i];
        }
        chunk_sums[tid] = prefixSum[chunk_end - 1];
    } 
    else {
        chunk_sums[tid] = 0;
    }

    #pragma omp barrier

    #pragma omp single
    {
        long long sum_so_far = 0;
        for (int i = 0; i < nthreads; i++) {
            long long temp = chunk_sums[i];
            chunk_sums[i] = sum_so_far;
            sum_so_far += temp;
        }
    }

    if (chunk_size > 0) {
        long long offset = chunk_sums[tid];
        for (long long i = chunk_start; i < chunk_end; i++) {
            prefixSum[i] += offset;
        }
    }

    #pragma omp single
    free(chunk_sums);
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