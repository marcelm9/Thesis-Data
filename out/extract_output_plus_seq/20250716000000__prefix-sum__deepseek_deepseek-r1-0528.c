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

  char *endptr;
  long long vector_size = strtoll(argv[1], &endptr, 10);
  if (*endptr != '\0' || vector_size <= 0) {
    printf("Invalid vector size\n");
    return 1;
  }

  long long seed = strtoll(argv[2], NULL, 10);
  srand((unsigned int)seed);

  int *arr = malloc(vector_size * sizeof(int));
  long long *prefixSum = malloc(vector_size * sizeof(long long));
  if (!arr || !prefixSum) {
    printf("Memory allocation failed\n");
    return 1;
  }

  for (long long i = 0; i < vector_size; i++) {
    arr[i] = rand() % 100;
  }

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  long long *chunk_sums = NULL, *chunk_offsets = NULL;
#pragma omp parallel shared(chunk_sums, chunk_offsets)
  {
    int tid = omp_get_thread_num();
    int num_threads = omp_get_num_threads();

#pragma omp single
    {
      chunk_sums = malloc(num_threads * sizeof(long long));
      chunk_offsets = malloc(num_threads * sizeof(long long));
      if (!chunk_sums || !chunk_offsets) {
        printf("Parallel memory allocation failed\n");
        exit(1);
      }
    }

    long long chunk_size = vector_size / num_threads;
    long long start_idx = tid * chunk_size;
    long long end_idx = (tid == num_threads - 1) ? vector_size : start_idx + chunk_size;

    long long local_sum = 0;
    if (start_idx < vector_size) {
      prefixSum[start_idx] = arr[start_idx];
      local_sum = arr[start_idx];
      for (long long i = start_idx + 1; i < end_idx; i++) {
        prefixSum[i] = prefixSum[i - 1] + arr[i];
        local_sum += arr[i];
      }
    }
    chunk_sums[tid] = local_sum;

#pragma omp barrier

#pragma omp single
    {
      chunk_offsets[0] = 0;
      for (int i = 1; i < num_threads; i++) {
        chunk_offsets[i] = chunk_offsets[i - 1] + chunk_sums[i - 1];
      }
    }

    if (start_idx < vector_size) {
      long long offset_val = chunk_offsets[tid];
      prefixSum[start_idx] += offset_val;
      for (long long i = start_idx + 1; i < end_idx; i++) {
        prefixSum[i] += offset_val;
      }
    }

#pragma omp barrier

#pragma omp single
    {
      free(chunk_sums);
      free(chunk_offsets);
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &end);
  double runtime = (end.tv_sec - start.tv_sec) +
                   (end.tv_nsec - start.tv_nsec) / 1000000000.0;

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