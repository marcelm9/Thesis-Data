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

  // compute prefix sum
  // This is a classic prefix sum problem which has data dependencies.
  // A direct parallelization of the sequential loop using #pragma omp for
  // would lead to race conditions due to `prefixSum[i - 1]` access.
  // One common parallel approach for prefix sum is a two-pass algorithm (Brent's algorithm/parallel scan).
  // For simplicity and to demonstrate a basic parallel approach while
  // still being correct (though not optimal for all cases),
  // we can use a parallel reduction followed by a sequential scan, or a more complex parallel scan.
  // Given the problem's nature, a simple direct parallelization of this specific kind of prefix sum
  // isn't straightforward without changing the algorithm fundamentally.
  // However, OpenMP 4.0 introduced the `scan` directive, but it's not universally supported or simple for this.
  // A common method for parallel prefix sum involves a sum-up phase and a down-sweep phase.

  // Using a parallel algorithm for prefix sum (e.g., using a work-efficient parallel scan like Blelloch's algorithm)
  // is significantly more complex than adding a simple #pragma omp for.
  // For cases where `vector_size` is large and performance is critical,
  // a dedicated parallel scan algorithm is needed.
  // For pedagogical purposes, if the intent is to show *some* OpenMP usage here,
  // while acknowledging the sequential bottleneck in the prefix sum calculation itself,
  // we can use OpenMP for other parts (like fill, which is already sequential here).
  // The provided sequential prefix sum loop is inherently sequential due to data dependencies.

  // A common trick for OpenMP prefix sum (often called parallel scan or work-efficient parallel prefix sum)
  // involves two passes. This can be complex to implement efficiently from scratch.
  // For this problem, if a direct parallelization of the `for` loop is intended,
  // it implies a fundamental change to the algorithm.
  // If the prompt implies parallelizing the *given* loop structure, it requires
  // a very advanced OpenMP technique or a restructuring of the algorithm.

  // Let's implement a common parallel prefix sum approach often seen with OpenMP,
  // which divides the array, computes local prefix sums, and then combines them.

  int num_threads;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }

  // Step 1: Calculate partial sums for each block
  long long *block_sums = malloc(num_threads * sizeof(long long));
  if (!block_sums) {
    printf("Memory allocation failed\n");
    return 1;
  }

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    long long start_idx = tid * (vector_size / num_threads);
    long long end_idx = (tid == num_threads - 1) ? vector_size : (tid + 1) * (vector_size / num_threads);

    if (start_idx < end_idx) { // Ensure block is not empty
      prefixSum[start_idx] = arr[start_idx];
      for (long long i = start_idx + 1; i < end_idx; ++i) {
        prefixSum[i] = prefixSum[i - 1] + arr[i];
      }
      block_sums[tid] = prefixSum[end_idx - 1]; // Store the last sum of the block
    } else {
        block_sums[tid] = 0; // Empty block
    }
  }

  // Step 2: Compute prefix sum of block_sums (sequential)
  for (int i = 1; i < num_threads; ++i) {
    block_sums[i] += block_sums[i - 1];
  }

  // Step 3: Add the accumulated block sums to subsequent blocks
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    if (tid > 0) { // All blocks except the first one need adjustment
      long long start_idx = tid * (vector_size / num_threads);
      long long end_idx = (tid == num_threads - 1) ? vector_size : (tid + 1) * (vector_size / num_threads);
      long long offset = block_sums[tid - 1];

      for (long long i = start_idx; i < end_idx; ++i) {
        prefixSum[i] += offset;
      }
    }
  }

  free(block_sums);

  // For `prefixSum[0] = arr[0];` which must be done at the start
  // This line is inherently sequential and critical for correctness.
  // In the parallel implementation above, the first element of each block calculates its own
  // prefix sum. The `prefixSum[0]` is correctly handled by thread 0.
  // The original sequential version:
  // prefixSum[0] = arr[0];
  // for (long long i = 1; i < vector_size; i++) {
  //   prefixSum[i] = prefixSum[i - 1] + arr[i];
  // }


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