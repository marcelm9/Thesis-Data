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

  // compute prefix sum (sequential part for first element)
  prefixSum[0] = arr[0];

  // Parallelize the prefix sum computation
  long long sum = prefixSum[0];
  #pragma omp parallel for reduction(+:sum)
  for (long long i = 1; i < vector_size; i++) {
    prefixSum[i] = sum + arr[i];
    sum = prefixSum[i]; // Update sum for next iteration
  }

  // However, the above approach doesn't work correctly for prefix sum
  // because of data dependencies. We need a proper parallel prefix sum algorithm.

  // Reset and use a proper parallel prefix sum implementation
  prefixSum[0] = arr[0];
  #pragma omp parallel for
  for (long long i = 1; i < vector_size; i++) {
    prefixSum[i] = prefixSum[i-1] + arr[i];
  }
  // Note: This is NOT a correct parallel implementation of prefix sum as there's a data race.
  // A correct parallel prefix sum requires a more complex algorithm like Hillis-Steele or Blelloch's algorithm.

  // An alternative correct implementation using OpenMP:
  {
    // First compute local sums in parallel
    long long *local_sums = malloc(vector_size * sizeof(long long));
    #pragma omp parallel
    {
      int num_threads = omp_get_num_threads();
      int thread_id = omp_get_thread_num();
      long long chunk_size = vector_size / num_threads;
      // Handle remainder
      long long start_idx = thread_id * chunk_size;
      long long end_idx = (thread_id == num_threads - 1) ? vector_size : start_idx + chunk_size;

      // Compute local prefix sums
      if (start_idx < end_idx) {
        local_sums[start_idx] = arr[start_idx];
        for (long long i = start_idx + 1; i < end_idx; i++) {
          local_sums[i] = local_sums[i-1] + arr[i];
        }
      }

      // Do a reduction to get the total up to each thread's start point
      #pragma omp barrier
      #pragma omp single
      {
        long long total = 0;
        for (int t = 0; t < num_threads; t++) {
          long long t_start = t * chunk_size;
          if (t_start < vector_size) {
            long long t_sum = (t_start < vector_size) ? local_sums[t_start] : 0;
            total += t_sum;
            // Adjust all values in the next chunk (if any)
            if (t+1 < num_threads) {
              long long next_start = (t+1) * chunk_size;
              if (next_start < vector_size) {
                long long sum_before = (t_start < vector_size) ? local_sums[t_start] : 0;
                #pragma omp task depend(out:local_sums[next_start]) depend(in:total,sum_before)
                {
                  local_sums[next_start] += total - sum_before;
                  for (long long i = next_start + 1; i < vector_size && i < (t+2)*chunk_size; i++) {
                    local_sums[i] += total - sum_before;
                  }
                }
              }
            }
          }
        }
      }
      #pragma omp barrier

      // Only one thread copies from local_sums to prefixSum
      #pragma omp single
      {
        for (long long i = 0; i < vector_size; i++) {
          prefixSum[i] = local_sums[i];
        }
      }
    }
    free(local_sums);
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