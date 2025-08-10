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

  // compute prefix sum using OpenMP parallel scan (Blelloch prefix sum)
  if (vector_size > 0) {
    // Up-sweep (reduce) phase
    #pragma omp parallel
    {
      for (long long h = 1; h < vector_size; h *= 2) {
        #pragma omp for
        for (long long i = 0; i < vector_size; i += 2 * h) {
          if (i + h < vector_size) {
            arr[i + 2 * h - 1] += arr[i + h - 1];
          }
        }
      }
    }

    // Down-sweep (distribute) phase
    // The last element of arr now holds the total sum
    if (vector_size > 0) {
        prefixSum[vector_size - 1] = arr[vector_size - 1]; // This is the total sum.
        arr[vector_size - 1] = 0; // Set the last element to 0 for the down-sweep
    }


    #pragma omp parallel
    {
      for (long long h = vector_size / 2; h >= 1; h /= 2) {
        #pragma omp for
        for (long long i = 0; i < vector_size; i += 2 * h) {
          long long t = arr[i + h - 1]; // Store original value of arr[i+h-1]
          arr[i + h - 1] = arr[i + 2 * h - 1];
          // Since prefix sum needs to be exclusive, we need the sum of elements before current index.
          // For inclusive, it would be prefixSum[i + 2 * h - 1] += t;
          arr[i + 2 * h - 1] += t;
        }
      }
    }
     // Copy the results to prefixSum. Note that this parallel scan modifies arr internally.
    #pragma omp parallel for
    for(long long i = 0; i < vector_size; ++i) {
        prefixSum[i] = arr[i];
    }
  }


  // end measuring
  clock_gettime(CLOCK_MONOTONIC, &end);
  double runtime = (end.tv_sec - start.tv_sec) +
                   (end.tv_nsec - start.tv_nsec) / 1000000000.0;

  // Re-fill the array with original values for verification purposes
  // This is a requirement for verification since the parallel scan modifies 'arr'.
  // In a real application, you might use a temporary array for the scan or modify the algorithm
  // if you need the original 'arr' contents.
  srand((unsigned int)seed); // Re-seed to generate the same random numbers
  for (long long i = 0; i < vector_size; i++) {
    arr[i] = rand() % 100;
  }

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