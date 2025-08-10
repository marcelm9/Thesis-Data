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

  // compute prefix sum using OpenMP parallel scan (Blelloch, 1990)
  // This is a two-pass algorithm.

  // Pass 1: Up-sweep (reduce)
  #pragma omp parallel
  {
    for (int d = 0; (1 << d) < vector_size; ++d) {
      #pragma omp for
      for (long long k = 0; k < vector_size; k += (1 << (d + 1))) {
        if (k + (1 << (d + 1)) - 1 < vector_size) { // Ensure within bounds
          arr[k + (1 << (d + 1)) - 1] += arr[k + (1 << d) - 1];
        }
      }
    }
  }

  prefixSum[vector_size - 1] = arr[vector_size - 1]; // Store total sum

  // Pass 2: Down-sweep (scan)
  #pragma omp parallel
  {
    for (int d = (int)(snprintf(NULL, 0, "%lld", vector_size - 1) * sizeof(char) * 8 / 3 + 1); d >= 0; --d) { // Log2 for vector_size
      #pragma omp for
      for (long long k = 0; k < vector_size; k += (1 << (d + 1))) {
        if (d > 0 && k + (1 << d) - 1 < vector_size && k + (1 << (d+1)) -1 < vector_size) {
            long long temp = arr[k + (1 << d) - 1];
            arr[k + (1 << d) - 1] = arr[k + (1 << (d+1)) - 1];
            arr[k + (1 << (d+1)) - 1] += temp;
        } else if (d == 0 && k < vector_size) { // Base case for scan_exclusive
            // For the OpenMP version of prefix sum using Blelloch's algorithm,
            // the result is accumulated in 'arr'.
            // The final prefix sums need to be extracted carefully.
            // If the original 'arr' is needed later, a copy should be made before the parallel scan.
        }
      }
    }
  }

  // Reconstruct prefixSum from the modified arr
  // This step can also be part of the parallel region if done carefully,
  // but for clarity and simplicity after the Blelloch scan,
  // we copy the result back to prefixSum.
  // Note: Blelloch's algorithm can yield an exclusive or inclusive scan depending on implementation.
  // Here, we adapt it to produce an inclusive scan similar to the serial version.
  #pragma omp parallel for
  for(long long i = 0; i < vector_size; ++i) {
      prefixSum[i] = arr[i];
  }


  // The above Blelloch parallel scan modifies 'arr' in-place.
  // To get the exact same `prefixSum` as the serial version which is `prefixSum[i] = prefixSum[i - 1] + arr[i]`,
  // a different parallel scan approach like "parallel prefix sum with reduction" or "segmented scan" might be more direct.
  // However, since the prompt asks for *parallelization* using OpenMP, and Blelloch's is a common parallel scan,
  // I will correct the values to match the serial output AFTER the parallel Blelloch scan.
  // This post-processing is what actually reconstructs the traditional inclusive prefix sum from the Blelloch output.

  // Let's re-run the `compute prefix sum` block with a more straightforward OpenMP parallel scan approach (associative scan).
  // A simple parallelization of the serial loop is not straightforward due to data dependencies.
  // Therefore, a common pattern for parallel prefix sum is a two-pass algorithm.
  // Or, for the purpose of matching the exact serial output without complex algorithm changes,
  // we can use a "block-wise" approach if the number of threads is small,
  // or a more robust parallel scan algorithm.

  // Let's re-implement the prefix sum with a parallel approach that more directly matches the serial output logic,
  // which might involve a less optimized but conceptually simpler parallelization.
  // A `std::exclusive_scan` or `std::inclusive_scan` equivalent in OpenMP.

  // Re-initialize for a simpler parallel approach that might be more intuitive
  // if the Blelloch implementation needs complex reconstruction to match the exact serial definition.
  // The simplest (though not most performant) way to parallelize this is
  // to calculate partial sums and then sum them up.

  // Let's go with a more robust parallel prefix sum for non-trivial sizes.
  // This uses a two-pass approach similar to Blelloch's but often explained as:
  // 1. Each thread computes a local prefix sum for its chunk.
  // 2. Compute prefix sum of the last elements of each chunk.
  // 3. Add these "block sums" to the local prefix sums.

  // Re-fill the array if the previous Blelloch scan modified it too much for this new approach
  // (The prompt implies maintaining `arr` as original for validation later).
  // I will re-`malloc` to ensure a clean state if the previous attempt changed `arr`.
  free(arr);
  free(prefixSum);
  arr = malloc(vector_size * sizeof(int));
  prefixSum = malloc(vector_size * sizeof(long long));
  if (!arr || !prefixSum) {
    printf("Memory allocation failed\n");
    return 1;
  }
  srand((unsigned int)seed); // Re-seed for consistent values
  for (long long i = 0; i < vector_size; i++) {
    arr[i] = rand() % 100; // random values in [0, 99]
  }

  clock_gettime(CLOCK_MONOTONIC, &start);

  // Parallel Prefix Sum (Two-Pass Approach)
  int num_threads = 0;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }

  // Step 1: Each thread computes its local prefix sum.
  // We need an array to store the sum of previous blocks for corrections.
  long long *block_sums = calloc(num_threads, sizeof(long long));
  if (!block_sums) {
      printf("Memory allocation failed\n");
      return 1;
  }

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    long long local_sum = 0;
    long long start_idx = (vector_size / num_threads) * tid;
    long long end_idx = (tid == num_threads - 1) ? vector_size : (vector_size / num_threads) * (tid + 1);

    for (long long i = start_idx; i < end_idx; ++i) {
      local_sum += arr[i];
      prefixSum[i] = local_sum;
    }
    block_sums[tid] = local_sum; // Store the total sum of this block
  }

  // Step 2: Compute prefix sum of block_sums (sequentially or in parallel if many threads).
  // Since num_threads is usually small, sequential is fine.
  for (int i = 1; i < num_threads; ++i) {
    block_sums[i] += block_sums[i - 1];
  }

  // Step 3: Add the appropriate block_sum to each element's prefixSum.
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    long long start_idx = (vector_size / num_threads) * tid;
    long long end_idx = (tid == num_threads - 1) ? vector_size : (vector_size / num_threads) * (tid + 1);

    long long prev_block_sum = (tid == 0) ? 0 : block_sums[tid - 1];

    for (long long i = start_idx; i < end_idx; ++i) {
      // For elements not in the first block, add the sum of all preceding blocks.
      // The prefixSum[i] already contains the local sum within its block.
      // E.g., for block 1, prefixSum[i] (already calculated) + block_sums[0]
      if (tid > 0) {
        prefixSum[i] += prev_block_sum;
      }
    }
  }

  free(block_sums);

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