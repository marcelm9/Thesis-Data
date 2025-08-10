#define COMPILERFLAGS "-lm -fopenmp"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define TOL 0.0001

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <grid_size> <steps>\n", argv[0]);
    return 1;
  }

  char *endptr;
  long N = strtol(argv[1], &endptr, 10);
  if (*endptr != '\0' || N <= 2) {
    printf("Invalid grid size\n");
    return 1;
  }

  long STEPS = strtol(argv[2], &endptr, 10);
  if (*endptr != '\0' || STEPS <= 0) {
    printf("Invalid number of steps\n");
    return 1;
  }

  // allocate grids
  double **grid = malloc(N * sizeof(double *));
  double **new_grid = malloc(N * sizeof(double *));
  double **ref_grid = malloc(N * sizeof(double *));

  if (!grid || !new_grid || !ref_grid) {
    printf("Memory allocation failed\n");
    return 1;
  }

  for (long i = 0; i < N; i++) {
    grid[i] = calloc(N, sizeof(double));
    new_grid[i] = calloc(N, sizeof(double));
    ref_grid[i] = calloc(N, sizeof(double));
    if (!grid[i] || !new_grid[i] || !ref_grid[i]) {
      printf("Memory allocation failed\n");
      return 1;
    }
  }

  // initialize with a central heat source
  grid[N / 2][N / 2] = 100.0;
  ref_grid[N / 2][N / 2] = 100.0;

  // run reference simulation
  for (long t = 0; t < STEPS; t++) {
    for (long i = 1; i < N - 1; i++) {
      for (long j = 1; j < N - 1; j++) {
        new_grid[i][j] = 0.25 * (ref_grid[i + 1][j] + ref_grid[i - 1][j] +
                                 ref_grid[i][j + 1] + ref_grid[i][j - 1]);
      }
    }
    // swap grids
    for (long i = 1; i < N - 1; i++)
      for (long j = 1; j < N - 1; j++)
        ref_grid[i][j] = new_grid[i][j];
  }

  // re-initialize main grid for timing
  for (long i = 0; i < N; i++)
    for (long j = 0; j < N; j++)
      grid[i][j] = 0.0;
  grid[N / 2][N / 2] = 100.0;

  // start timing
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  for (long t = 0; t < STEPS; t++) {
    #pragma omp parallel for private(i, j)
    for (long i = 1; i < N - 1; i++) {
      for (long j = 1; j < N - 1; j++) {
        new_grid[i][j] = 0.25 * (grid[i + 1][j] + grid[i - 1][j] +
                                 grid[i][j + 1] + grid[i][j - 1]);
      }
    }
    // swap grids
    #pragma omp parallel for private(i, j)
    for (long i = 1; i < N - 1; i++)
      for (long j = 1; j < N - 1; j++)
        grid[i][j] = new_grid[i][j];
  }

  clock_gettime(CLOCK_MONOTONIC, &end);
  double runtime = (end.tv_sec - start.tv_sec) +
                   (end.tv_nsec - start.tv_nsec) / 1000000000.0;

  // check results
  int err = 0;
  #pragma omp parallel for reduction(+:err) private(i, j)
  for (long i = 1; i < N - 1; i++) {
    for (long j = 1; j < N - 1; j++) {
      double diff = fabs(grid[i][j] - ref_grid[i][j]);
      if (diff > TOL)
        err++;
    }
  }

  printf("[%d, %f]\n", err, runtime);

  // free memory
  for (long i = 0; i < N; i++) {
    free(grid[i]);
    free(new_grid[i]);
    free(ref_grid[i]);
  }
  free(grid);
  free(new_grid);
  free(ref_grid);

  return 0;
}