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

  // fill the array in parallel
  #pragma omp parallel for
  for (long long i = 0; i < vector_size; i++) {
    // Use thread-specific random seeds to avoid race conditions
    unsigned int thread_seed = seed + omp_get_thread_num();
    arr[i] = rand_r(&thread_seed) % 100; // random values in [0, 99]
  }

  // start measuring
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  // compute prefix sum - this needs to be done in chunks
  prefixSum[0] = arr[0];
  
  // Parallel scan algorithm
  int num_threads;
  long long chunk_size;
  
  #pragma omp parallel
  {
    #pragma omp single
    {
      num_threads = omp_get_num_threads();
      chunk_size = (vector_size + num_threads - 1) / num_threads;
    }
  }
  
  long long *local_sums = malloc(num_threads * sizeof(long long));
  if (!local_sums) {
    printf("Memory allocation failed\n");
    free(arr);
    free(prefixSum);
    return 1;
  }
  
  // Phase 1: Local prefix sums
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    long long start_idx = tid * chunk_size;
    long long end_idx = (tid + 1) * chunk_size;
    if (end_idx > vector_size) end_idx = vector_size;
    
    if (start_idx < vector_size) {
      long long local_sum = arr[start_idx];
      prefixSum[start_idx] = local_sum;
      
      for (long long i = start_idx + 1; i < end_idx; i++) {
        local_sum += arr[i];
        prefixSum[i] = local_sum;
      }
      
      local_sums[tid] = local_sum;
    }
  }
  
  // Phase 2: Prefix sum on local sums
  for (int i = 1; i < num_threads; i++) {
    local_sums[i] += local_sums[i-1];
  }
  
  // Phase 3: Add the offsets to each chunk
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    if (tid > 0) {  // Skip the first chunk
      long long offset = local_sums[tid-1];
      long long start_idx = tid * chunk_size;
      long long end_idx = (tid + 1) * chunk_size;
      if (end_idx > vector_size) end_idx = vector_size;
      
      for (long long i = start_idx; i < end_idx; i++) {
        prefixSum[i] += offset;
      }
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
  free(local_sums);

  return 0;
}