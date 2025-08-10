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
    arr[i] = rand() % 100; // random values in [0, 99]
  }

  // start measuring
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  // Compute prefix sum using parallel scan
  prefixSum[0] = arr[0];
  
  // We can't fully parallelize a standard prefix sum due to dependencies,
  // but we can use a parallel scan (prefix sum) algorithm
  
  // First parallel step: compute local sums
  const int chunk_size = 1024; // or another suitable size
  long long *local_sums = NULL;
  int num_chunks = (vector_size + chunk_size - 1) / chunk_size;
  
  if (num_chunks > 1) {
    local_sums = malloc(num_chunks * sizeof(long long));
    if (!local_sums) {
      printf("Memory allocation failed for local sums\n");
      free(arr);
      free(prefixSum);
      return 1;
    }
    
    #pragma omp parallel for
    for (int c = 0; c < num_chunks; c++) {
      long long start_idx = c * chunk_size;
      long long end_idx = (c + 1) * chunk_size;
      if (end_idx > vector_size) end_idx = vector_size;
      
      long long local_sum = arr[start_idx];
      prefixSum[start_idx] = local_sum;
      
      for (long long i = start_idx + 1; i < end_idx; i++) {
        local_sum += arr[i];
        prefixSum[i] = local_sum;
      }
      
      local_sums[c] = local_sum;
    }
    
    // Compute prefix sums of chunk sums (this is sequential and usually small)
    for (int c = 1; c < num_chunks; c++) {
      local_sums[c] += local_sums[c - 1];
    }
    
    // Add chunk prefix sums to each element except the first chunk
    #pragma omp parallel for
    for (int c = 1; c < num_chunks; c++) {
      long long start_idx = c * chunk_size;
      long long end_idx = (c + 1) * chunk_size;
      if (end_idx > vector_size) end_idx = vector_size;
      
      long long offset = local_sums[c - 1];
      for (long long i = start_idx; i < end_idx; i++) {
        prefixSum[i] += offset;
      }
    }
    
    free(local_sums);
  } else {
    // If there's only one chunk, compute directly
    prefixSum[0] = arr[0];
    for (long long i = 1; i < vector_size; i++) {
      prefixSum[i] = prefixSum[i - 1] + arr[i];
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