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

  // parallel prefix sum with OpenMP (Blelloch scan)
  int num_threads = 1;
  #pragma omp parallel
  {
    #pragma omp single
    num_threads = omp_get_num_threads();
  }

  long long *partial_sums = malloc(num_threads * sizeof(long long));

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();

    long long chunk = vector_size / nthreads;
    long long rem = vector_size % nthreads;
    long long start_idx = tid * chunk + (tid < rem ? tid : rem);
    long long end_idx = start_idx + chunk + (tid < rem ? 1 : 0);

    // Step 1: Each thread computes local prefix sums
    if (end_idx > start_idx) {
      prefixSum[start_idx] = arr[start_idx];
      for (long long i = start_idx + 1; i < end_idx; i++) {
        prefixSum[i] = prefixSum[i - 1] + arr[i];
      }
      partial_sums[tid] = prefixSum[end_idx - 1];
    } else {
      partial_sums[tid] = 0;
    }

    #pragma omp barrier

    // Step 2: Calculate offsets for each thread
    long long offset = 0;
    for (int i = 0; i < tid; i++) {
      offset += partial_sums[i];
    }

    // Step 3: Add offset to local prefix sums (skip offset for first block)
    if (tid > 0 && end_idx > start_idx) {
      for (long long i = start_idx; i < end_idx; i++) {
        prefixSum[i] += offset;
      }
    }
  }

  free(partial_sums);

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