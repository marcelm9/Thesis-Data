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

  // Parallel computation using OpenMP
  long long *chunk_sums = NULL;
  long long *offsets_arr = NULL;

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    long long start_i = (vector_size * tid) / nthreads;
    long long end_i = (vector_size * (tid+1)) / nthreads;

    if (start_i < end_i) {
      prefixSum[start_i] = arr[start_i];
      for (long long i = start_i+1; i < end_i; i++) {
        prefixSum[i] = prefixSum[i-1] + arr[i];
      }
    }

    #pragma omp barrier

    #pragma omp single
    {
      chunk_sums = malloc(nthreads * sizeof(long long));
    }

    if (start_i < end_i) {
      chunk_sums[tid] = prefixSum[end_i-1];
    } else {
      chunk_sums[tid] = 0;
    }

    #pragma omp barrier

    #pragma omp single
    {
      offsets_arr = malloc(nthreads * sizeof(long long));
      offsets_arr[0] = 0;
      for (int idx = 1; idx < nthreads; idx++) {
        offsets_arr[idx] = offsets_arr[idx-1] + chunk_sums[idx-1];
      }
    }

    if (start_i < end_i) {
      long long offset_val = offsets_arr[tid];
      for (long long i = start_i; i < end_i; i++) {
        prefixSum[i] += offset_val;
      }
    }

    #pragma omp barrier

    #pragma omp single
    {
      free(chunk_sums);
      free(offsets_arr);
    }
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