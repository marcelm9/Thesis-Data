#define COMPILERFLAGS "-lm -fopenmp"
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TOL 0.00001
#define K 3

int main(int argc, char *argv[]) {
  if (argc != 3) {
    printf("Usage: %s <image_size> <seed>\n", argv[0]);
    return 1;
  }

  long N = strtol(argv[1], NULL, 10);
  if (N <= 2) {
    printf("Image size must be greater than 2\n");
    return 1;
  }

  srand((unsigned int)strtol(argv[2], NULL, 10));

  float **image = malloc(N * sizeof(float *));
  float **output = malloc(N * sizeof(float *));
  float **reference = malloc(N * sizeof(float *));
  for (int i = 0; i < N; i++) {
    image[i] = malloc(N * sizeof(float));
    output[i] = malloc(N * sizeof(float));
    reference[i] = malloc(N * sizeof(float));
  }

  if (!image || !output || !reference) {
    printf("Memory allocation failed\n");
    return 1;
  }

  float kernel[K][K] = {{1 / 16.0, 2 / 16.0, 1 / 16.0},
                        {2 / 16.0, 4 / 16.0, 2 / 16.0},
                        {1 / 16.0, 2 / 16.0, 1 / 16.0}};

  int multiplier = rand() % 1000;
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      image[i][j] = (float)(i + j) * multiplier / N;

  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++) {
      float sum = 0.0;
      for (int ki = 0; ki < K; ki++)
        for (int kj = 0; kj < K; kj++)
          sum += kernel[ki][kj] * image[i + ki - 1][j + kj - 1];
      reference[i][j] = sum;
    }
  }

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

#pragma omp parallel for collapse(2) schedule(static)
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++) {
      float sum = 0.0;
      for (int ki = 0; ki < K; ki++)
        for (int kj = 0; kj < K; kj++)
          sum += kernel[ki][kj] * image[i + ki - 1][j + kj - 1];
      output[i][j] = sum;
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &end);
  double runtime = (end.tv_sec - start.tv_sec) +
                   (end.tv_nsec - start.tv_nsec) / 1000000000.0;

  int errors = 0;
  for (int i = 1; i < N - 1; i++) {
    for (int j = 1; j < N - 1; j++) {
      if (fabs(output[i][j] - reference[i][j]) > TOL)
        errors++;
    }
  }

  printf("[%d, %f]\n", errors, runtime);

  for (int i = 0; i < N; i++) {
    free(image[i]);
    free(output[i]);
    free(reference[i]);
  }
  free(image);
  free(output);
  free(reference);

  return 0;
}