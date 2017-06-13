#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <stdint.h>
#include <string.h>

#ifndef IMG_SIZE
#define IMG_SIZE 1024*32-2
#endif

#ifndef NUM_THREADS
#define NUM_THREADS 8
#endif

typedef struct {
  unsigned int    width;
  unsigned int    height;
  unsigned int    internalWidth;
  unsigned int    internalHeight;
  uint8_t*        img;
} image_t;

void imgDestroy(image_t* im) {
  free(im->img);
  free(im);
}

void printMatrix_d(double* m, int width, int height) {
  int row, col;
  for(row=0; row < height; row++) {
    for(col=0; col < width; col++) {
      printf("%.1f\t", m[(row*height)+col]);
    }
    printf("\n");
  }
  return;
}

void printMatrix(FILE* fp, image_t* m) {
  int row, col;
  for(row=0; row < m->internalHeight; row++) {
    for(col=0; col < m->internalWidth; col++) {
      fprintf(fp, "%u\t", m->img[(row*(m->internalHeight))+col]);
    }
    fprintf(fp, "\n");
  }
  return;
}

image_t* alloc_img(int width, int height) {
  image_t* im;
  im = malloc(sizeof(image_t));

  im->width = width;
  im->height = height;

  // Extra rows and columns to make convolution a little easier.
  // IMPORTANT: since I'm adding only 2 pixels to each dimension,
  //            this will only work for 3x3 kernels!!!
  im->internalWidth = width+2;
  im->internalHeight = height+2;

  int size = (im->internalWidth) * (im->internalHeight) * sizeof(uint8_t);
  im->img = calloc(1, size);

  if (im->img == NULL) {
    fprintf(stderr, "Could not allocate image (size %ux%u)\n", width, height);
    exit(EXIT_FAILURE);
  }
  return im;
}

image_t* filter(image_t* im, double* K, int Ks, int divisor) {
  image_t* oi;
  oi = alloc_img(im->width, im->height);

  #pragma omp parallel num_threads(NUM_THREADS)
  {
  float acc;
  unsigned int irow, icol;
  int krow, kcol;
    int id = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    for(irow=(id+1); irow <= im->width; irow = irow + nthreads) {
      for(icol=1; icol <= im->height; icol++) {
        acc = 0;
        for(krow=-Ks; krow <= Ks; ++krow) {
          for(kcol=-Ks; kcol <= Ks; ++kcol) {
            acc += im->img[((irow+krow)*(im->internalHeight))+(icol+kcol)] * K[((Ks+krow)*(2*Ks+1)) + (Ks+kcol)] / divisor;
          }
        }
        // clamp
        oi->img[(irow*(oi->internalHeight))+icol] = acc > 255.0 ? 255 : (unsigned int)acc;
        oi->img[(irow*(oi->internalHeight))+icol] = acc < 0.0   ? 0   : (unsigned int)acc;
      }
    }
  }
  return oi;
}

// sample kernels
int    identityKernelDivisor = 1;
double identityKernel[3*3] = {0, 0, 0,
                              0, 1, 0,
                              0, 0, 0};

int    sharpenDivisor = 1;
double sharpenKernel[3*3] = { 0, -1,  0,
                             -1,  5, -1,
                              0, -1,  0};

int    edgeDetectionKernelDivisor = 1;
double edgeDetectionKernel[3*3] = {-1, -1, -1,
                                   -1,  8, -1,
                                   -1, -1, -1};

image_t* genRandomImage(unsigned int size) {
  image_t* im = alloc_img(size, size);
  for(int row=1; row <= im->height; ++row) {
    for(int col=1; col <= im->height; ++col) {
      im->img[(row*(im->internalHeight))+col] = rand() % 255;
    }
  }
  return im;
}

int main(int argc, char** argv) {
  srand(24);

  printf("Generating image...\n");
  image_t* original = genRandomImage(IMG_SIZE);

  printf("Running convolution...\n");
  printf("Kernel:\n");
  printMatrix_d(sharpenKernel, 3, 3);

  double start, end;
  start = omp_get_wtime();

  image_t* filtered = filter(original, sharpenKernel, 1, sharpenDivisor);

  end = omp_get_wtime();
  printf("%f\n", end-start);

  FILE* ofp = fopen("original.dat", "w");
  printMatrix(ofp, original);
  fclose(ofp);

  FILE* ffp = fopen("filtered.dat", "w");
  printMatrix(ffp, filtered);
  fclose(ffp);

  imgDestroy(original);
  imgDestroy(filtered);
  return 0;
}
