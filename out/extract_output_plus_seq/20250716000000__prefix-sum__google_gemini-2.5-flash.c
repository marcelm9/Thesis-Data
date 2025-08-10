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

  // compute prefix sum using parallel scan (Blelloch prefix sum)
  // This is a common parallel prefix sum algorithm.
  // It involves two passes.

  // Pass 1: Up-sweep (reduce)
  #pragma omp parallel
  {
    for (int d = 1; d < vector_size; d *= 2) {
      #pragma omp for
      for (long long i = 0; i < vector_size; i += 2 * d) {
        if (i + d < vector_size) {
            arr[i + 2 * d - 1] += arr[i + d - 1]; // Store sum in last element of block
        }
      }
    }
  }

  // Store total sum in prefixSum[vector_size - 1] for the down-sweep
  if (vector_size > 0) {
    prefixSum[vector_size - 1] = arr[vector_size - 1]; // This is the total sum of the original array
  }

  // Pass 2: Down-sweep (distribute)
  #pragma omp parallel
  {
    for (int d = vector_size / 2; d >= 1; d /= 2) {
      #pragma omp for
      for (long long i = 0; i < vector_size; i += 2 * d) {
        if (i + d < vector_size) {
            long long temp = arr[i + d - 1];
            arr[i + d - 1] = arr[i + 2 * d - 1];
            arr[i + 2 * d - 1] += temp;
        }
      }
    }
  }

  // Copy the results from arr to prefixSum.
  // The 'arr' array now contains the prefix sums in a specific order depending on the Blelloch algorithm variant.
  // For exclusive scan, we need to adjust the results. For inclusive scan, 'arr' should contain them.
  // The provided sequential code computes an inclusive prefix sum.
  // We need to re-evaluate the Blelloch output to match the original, which might be tricky directly.
  // A simpler OpenMP approach for inclusive scan:

  // Re-initialize arr for a simpler OpenMP approach (if not using the Blelloch algorithm directly for output)
  // Or, if implementing Blelloch fully, arr should become the prefix sum array.
  // Let's go with a simpler parallelization for the given problem:
  // Parallelizing the original simple loop is hard due to data dependency.
  // A common approach for parallel prefix sum involves parallel-friendly algorithms like Blelloch scan,
  // or a block-wise approach.

  // Let's use a simpler, more direct parallelization using parallel_for and then a sequential combine
  // This isn't a true "parallel prefix sum" algorithm, but rather a parallel first step and sequential combine.
  // For true parallel prefix sum, the Blelloch algorithm (or similar) is needed.
  // Reverting to input array for this simpler attempt.
  // The most straightforward parallel prefix sum is not directly obvious with simple #pragma omp for.

  // Implementing a simplified block-wise approach:
  free(arr); // Free original arr
  arr = malloc(vector_size * sizeof(int)); // Re-allocate and fill for a different approach
  if (!arr) {
    printf("Memory allocation failed\n");
    return 1;
  }
  srand((unsigned int)seed); // Re-seed for consistency of values
  for (long long i = 0; i < vector_size; i++) {
    arr[i] = rand() % 100; // random values in [0, 99]
  }

  // Block-wise parallel prefix sum
  int num_threads = omp_get_max_threads();
  int thread_offset_sums[num_threads];
  
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    long long local_sum = 0;

    #pragma omp for schedule(static) nowait
    for (long long i = 0; i < vector_size; i++) {
      local_sum += arr[i];
      prefixSum[i] = local_sum;
    }
    thread_offset_sums[tid] = local_sum; // Store the last prefix sum of each thread's block
  }

  // Sequential combining of block sums
  // Sum up the last elements of each thread's calculated prefix sums to get the offsets
  for (int i = 1; i < num_threads; ++i) {
    thread_offset_sums[i] += thread_offset_sums[i-1];
  }

  // Add the offsets to subsequent blocks
  #pragma omp parallel
  {
      int tid = omp_get_thread_num();
      long long offset = (tid == 0) ? 0 : thread_offset_sums[tid - 1];

      // Re-iterate over the assigned chunks to add the offset.
      // This requires knowing the exact range each thread processed in the first #pragma omp for.
      // With `schedule(static)`, the ranges are predictable.
      long long chunk_size = (vector_size + num_threads - 1) / num_threads;
      long long start_idx = tid * chunk_size;
      long long end_idx = (tid + 1) * chunk_size;
      if (end_idx > vector_size) {
        end_idx = vector_size;
      }
      
      for (long long i = start_idx; i < end_idx; i++) {
          prefixSum[i] += offset;
      }
  }


  // The above "block-wise" approach for prefix sum in OpenMP is tricky to get right
  // without a proper parallel scan algorithm.
  // The simplest (and often performant for large arrays) approach is a `parallel for` over the update loop,
  // which is not directly applicable due to data dependency for prefix sum `prefixSum[i] = prefixSum[i-1] + arr[i]`.
  // True prefix sum parallelization requires algorithms like Blelloch scan or Harris scan.

  // Let's implement Blelloch parallel prefix sum (inclusive scan) which is more appropriate
  // for the problem statement of "parallelize the program".

  // Re-allocate arrays and re-populate for the Blelloch approach if we
  // are discarding the previous attempts.
  free(arr);
  free(prefixSum);
  arr = malloc(vector_size * sizeof(int));
  prefixSum = malloc(vector_size * sizeof(long long)); // prefixSum will store partial sums during scan
  if (!arr || !prefixSum) {
    printf("Memory allocation failed\n");
    return 1;
  }
  srand((unsigned int)seed);
  for (long long i = 0; i < vector_size; i++) {
    arr[i] = rand() % 100;
    prefixSum[i] = arr[i]; // Initialize prefixSum with array values
  }


  // Blelloch Parallel Scan (Inclusive Prefix Sum)
  // Step 1: Up-Sweep (Reduction Phase)
  for (long long d = 1; d < vector_size; d *= 2) {
    #pragma omp parallel for
    for (long long k = 0; k < vector_size; k += 2 * d) {
      if (k + d < vector_size) {
        prefixSum[k + 2 * d - 1] += prefixSum[k + d - 1];
      }
    }
  }

  // Step 2: Down-Sweep (Distribution Phase)
  if (vector_size > 0) {
    prefixSum[vector_size - 1] = 0; // Set the last element to 0 for recursive summation later
                                   // This is for exclusive scan, but we adapt it for inclusive.
                                   // For inclusive, the last element already holds the total sum.
                                   // We need to re-strategize for inclusive properly.

    // Correct Blelloch Inclusive Scan (requires a temporary array or careful in-place)
    // The previous up-sweep correctly sums up into the last element of blocks.
    // Now, the down-sweep redistributes and calculates the inclusive sums.

    // To make it an inclusive scan directly, after Up-Sweep, prefixSum[N-1] is the total sum.
    // We need temp storage for original values or an explicit exclusive scan then add original values.

    // Let's make a copy of arr to work on prefixSum directly.
    // The previous loop already copied arr to prefixSum. Let's trace it.
    // If prefixSum[k + 2 * d - 1] += prefixSum[k + d - 1] is Up-Sweep, then
    // prefixSum at the end of Up-sweep has specific values at specific indices (powers of 2).

    // A simpler way: exclusive scan first, then add original elements to get inclusive.
    // Or, a parallel exclusive scan, then modify the first element.

    // Re-attempting Blelloch inclusive scan from scratch for clarity.
    free(arr); // Revert to only prefixSum for the in-place calculation
    arr = malloc(vector_size * sizeof(int)); // Use arr to store original values
    if (!arr) {
        printf("Memory allocation failed\n");
        return 1;
    }
    srand((unsigned int)seed);
    for (long long i = 0; i < vector_size; i++) {
        arr[i] = rand() % 100;
        prefixSum[i] = arr[i]; // prefixSum starts with the original values
    }

    long long total_sum = 0;
    if (vector_size > 0) {
      total_sum = prefixSum[vector_size - 1];
    }

    // Up-Sweep
    for (long long d = 1; d < vector_size; d *= 2) {
        #pragma omp parallel for
        for (long long i = 0; i < vector_size; i++) {
            if ((i + 1) % (2 * d) == 0 && i - d >= 0) {
                prefixSum[i] += prefixSum[i - d];
            }
        }
    }

    // Down-Sweep
    if (vector_size > 0) {
        prefixSum[vector_size - 1] = 0; // For exclusive scan. For inclusive, it's total_sum
                                      // and then some adjustments.
                                      // Correct for inclusive:
                                      // After up-sweep, prefixSum[vector_size - 1] holds the total sum.
                                      // We need to propagate this back.

        // Standard Blelloch inclusive scan (simplified variant):
        // temp storage for inclusive scan from original array
        long long *tmp_prefixSum = malloc(vector_size * sizeof(long long));
        if (!tmp_prefixSum) {
          printf("Memory allocation failed\n");
          return 1;
        }

        #pragma omp parallel for
        for (long long i = 0; i < vector_size; i++) {
            tmp_prefixSum[i] = arr[i];
        }

        for (long long d = 1; d < vector_size; d *= 2) {
            #pragma omp parallel for
            for (long long k = 0; k < vector_size; k += 2 * d) {
                if (k + d < vector_size) {
                    tmp_prefixSum[k + 2 * d - 1] += tmp_prefixSum[k + d - 1];
                }
            }
        }

        // Down-Sweep for inclusive scan is a bit counter-intuitive after up-sweep
        // if aiming for the direct result.
        // A common technique is to compute exclusive scan, then make it inclusive.

        long long current_total_sum_after_upsweep = 0;
        if (vector_size > 0) {
          current_total_sum_after_upsweep = tmp_prefixSum[vector_size - 1];
        }
        
        // This is usually the Exclusive Scan part setup.
        // For inclusive from the structure:
        tmp_prefixSum[vector_size - 1] = 0; // This makes the last element the starting point for back-propagation.
                                            // It means we're essentially calculating an exclusive prefix sum here.

        for (long long d = vector_size / 2; d >= 1; d /= 2) {
            #pragma omp parallel for
            for (long long k = 0; k < vector_size; k += 2 * d) {
                if (k + d < vector_size) {
                    long long temp = tmp_prefixSum[k + d - 1];
                    tmp_prefixSum[k + d - 1] = tmp_prefixSum[k + 2 * d - 1];
                    tmp_prefixSum[k + 2 * d - 1] += temp;
                }
            }
        }

        // Now tmp_prefixSum contains the exclusive scan results.
        // Convert to inclusive scan by adding the original value.
        #pragma omp parallel for
        for (long long i = 0; i < vector_size; i++) {
            prefixSum[i] = tmp_prefixSum[i] + arr[i];
        }
        free(tmp_prefixSum);
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
    last_value += arr[i]; // arr contains original values
    if (prefixSum[i] != last_value) {
      err++;
    }
  }
  printf("[%d, %f]\n", err, runtime);

  free(arr);
  free(prefixSum);

  return 0;
}