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

  long long *chunk_totals = NULL;
  long long *offsets = NULL;

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int num_threads = omp_get_num_threads();
    long long chunk_size = (vector_size + num_threads - 1) / num_threads;
    long long start_index = tid * chunk_size;
    long long end_index = start_index + chunk_size;
    if (end_index > vector_size) {
      end_index = vector_size;
    }

#pragma omp single
    {
      chunk_totals = malloc(num_threads * sizeof(long long));
      offsets = malloc(num_threads * sizeof(long long));
    }

#pragma omp barrier

    long long local_sum = 0;
    if (start_index < vector_size) {
      prefixSum[start_index] = arr[start_index];
      local_sum = arr[start_index];
      for (long long i = start_index + 1; i < end_index; i++) {
        prefixSum[i] = prefixSum[i - 1] + arr[i];
        local_sum += arr[i];
      }
    }
    chunk_totals[tid] = local_sum;

#pragma omp barrier

#pragma omp single
    {
      offsets[0] = 0;
      for (int i = 1; i < num_threads; i++) {
        offsets[i] = offsets[i - 1] + chunk_totals[i - 1];
      }
    }

    if (start_index < vector_size) (long long i (long long i = start_index; i < end_index; i++) {
        prefixSum[i] += offsets[tid];
      }
    }
  }

  free(chunk_totals);
  free(offsets);

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