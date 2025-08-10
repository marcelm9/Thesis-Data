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
    arr[i] = rand() % 100;
  }

  // start measuring
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  long long *chunk_sums = NULL;
  #pragma omp parallel
  {
    int num_t = omp_get_num_threads();
    int tid = omp_get_thread_num();

    long long start_idx = (vector_size * tid) / num_t;
    long long end_idx = (vector_size * (tid + 1)) / num_t;
    long long chunk_size = end_idx - start_idx;

    if (chunk_size > 0) {
      prefixSum[start_idx] = arr[start_idx];
      for (long long j = start_idx + 1; j < end_idx; j++) {
        prefixSum[j] = prefixSum[j - 1] + arr[j];
      }
    }

    long long my_chunk_total = (chunk_size > 0) ? prefixSum[end_idx - 1] : 0;

    #pragma omp barrier

    #pragma omp single
    {
      chunk_sums = (long long *)malloc(num_t * sizeof(long long));
    }

    chunk_sums[tid] = my_chunk_total;

    #pragma omp barrier

    #pragma omp single
    {
      long long accum = 0;
      for (int i = 0; i < num_t; i++) {
        long long temp = chunk_sums[i];
        chunk_sums[i] = accum;
        accum += temp;
      }
    }

    #pragma omp barrier

    if (chunk_size > 0) {
      long long offset = chunk_sums[tid];
      for (long long j = start_idx; j < end_idx; j++) {
        prefixSum[j] += offset;
      }
    }
  }

  free(chunk_sums);

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