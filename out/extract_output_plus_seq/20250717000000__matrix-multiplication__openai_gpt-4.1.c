#define COMPILERFLAGS "-fopenmp -lm"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define TOL 0.0000001

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <matrix_size> <seed>\n", argv[0]);
    return 1;
  }

  srand((unsigned int)strtol(argv[2], NULL, 10));

  char *endptr;
  long n = strtol(argv[1], &endptr, 10);
  if (*endptr != '\0' || n <= 0) {
    printf("Invalid matrix size\n");
    return 1;
  }

  double **A = malloc(n * sizeof(double *));
  double **B = malloc(n * sizeof(double *));
  double **C = malloc(n * sizeof(double *));
  double **Correct = malloc(n * sizeof(double *));
  int err = 0;

  for (int i = 0; i < n; i++) {
    A[i] = malloc(n * sizeof(double));
    B[i] = malloc(n * sizeof(double));
    C[i] = malloc(n * sizeof(double));
    Correct[i] = malloc(n * sizeof(double));
  }

  if (!A || !B || !C) {
    printf("Memory allocation failed\n");
    return 1;
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      A[i][j] = (double)rand() / RAND_MAX;
      B[i][j] = (double)rand() / RAND_MAX;
      C[i][j] = 0.0;
    }
  }

  // create correct solution
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      Correct[i][j] = 0.0;
      for (int k = 0; k < n; k++) {
        Correct[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  // Parallelize this loop with OpenMP
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &end);
  double runtime = (end.tv_sec - start.tv_sec) +
                   (end.tv_nsec - start.tv_nsec) / 1000000000.0;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double val = fabs(C[i][j] - Correct[i][j]);
      if (val > TOL)
        err++;
    }
  }

  printf("[%d, %f]\n", err, runtime);

  for (int i = 0; i < n; i++) {
    free(A[i]);
    free(B[i]);
    free(C[i]);
    free(Correct[i]);
  }
  free(A);
  free(B);
  free(C);
  free(Correct);

  return 0;
}