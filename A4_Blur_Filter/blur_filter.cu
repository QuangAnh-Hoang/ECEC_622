/* Reference code implementing the box blur filter.

    Build and execute as follows: 
        make clean && make 
        ./blur_filter size

    Author: Naga Kandasamy
    Date created: May 3, 2019
    Date modified: May 12, 2020

    FIXME: Quang Anh Hoang
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <float.h>

/* #define DEBUG */

/* Include the kernel code */
#include "blur_filter_kernel.cu"

extern "C" void compute_gold(const image_t, image_t);
void compute_on_device(const image_t, image_t);
int check_results(const float *, const float *, int, float);
void print_image(const image_t);

#define THREAD_PER_BLOCK 128;
#define NUM_BLOCKS 240;

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s size\n", argv[0]);
        fprintf(stderr, "size: Height of the image. The program assumes size x size image.\n");
        exit(EXIT_FAILURE);
    }

    /* Allocate memory for the input and output images */
    int size = atoi(argv[1]);

    fprintf(stderr, "Creating %d x %d images\n", size, size);
    image_t in, out_gold, out_gpu;
    in.size = out_gold.size = out_gpu.size = size;
    in.element = (float *)malloc(sizeof(float) * size * size);
    out_gold.element = (float *)malloc(sizeof(float) * size * size);
    out_gpu.element = (float *)malloc(sizeof(float) * size * size);
    if ((in.element == NULL) || (out_gold.element == NULL) || (out_gpu.element == NULL)) {
        perror("Malloc");
        exit(EXIT_FAILURE);
    }

    /* Poplulate our image with random values between [-0.5 +0.5] */
    srand(time(NULL));
    int i;
    for (i = 0; i < size * size; i++)
        in.element[i] = rand()/(float)RAND_MAX -  0.5;
  
   /* Calculate the blur on the CPU. The result is stored in out_gold. */
    fprintf(stderr, "Calculating blur on the CPU\n"); 
    compute_gold(in, out_gold); 

#ifdef DEBUG 
   print_image(in);
   print_image(out_gold);
#endif

   /* FIXME: Calculate the blur on the GPU. The result is stored in out_gpu. */
   fprintf(stderr, "Calculating blur on the GPU\n");
   compute_on_device(in, out_gpu);

   /* Check CPU and GPU results for correctness */
   fprintf(stderr, "Checking CPU and GPU results\n");
   int num_elements = out_gold.size * out_gold.size;
   float eps = 1e-6;    /* Do not change */
   int check;
   check = check_results(out_gold.element, out_gpu.element, num_elements, eps);
   if (check == 0) 
       fprintf(stderr, "TEST PASSED\n");
   else
       fprintf(stderr, "TEST FAILED\n");

//   print_image(in);
//   print_image(out_gpu);
   
   /* Free data structures on the host */
   free((void *)in.element);
   free((void *)out_gold.element);
   free((void *)out_gpu.element);

    exit(EXIT_SUCCESS);
}

/* FIXME: Complete this function to calculate the blur on the GPU */
void compute_on_device(const image_t in, image_t out)
{
    int num_elements = in.size * in.size;

    float *in_on_device = NULL;
    float *out_on_device = NULL;

    int size = in.size;
    out.size = size;

    /* Allocate space on GPU for vector in and transfer from host to GPU */
    cudaMalloc((void **)&in_on_device, num_elements * sizeof(float));
    cudaMemcpy(in_on_device, in.element, num_elements * sizeof(float), cudaMemcpyHostToDevice);

    /* Allocate space on GPU for result vector */
    cudaMalloc((void **)&out_on_device, num_elements * sizeof(float));

//    int num_block = NUM_BLOCKS;
//    dim3 thread_block(THREAD_PER_BLOCK, 1, 1);
//    fprintf(stderr, "Setting up a (%d x 1) execution grid\n", num_block);
//    dim3 grid(num_block, 1);

    fprintf(stderr, "Applying blur filter using GPU\n");

    int num_blocks = num_elements/128;

    blur_filter_kernel<<< num_blocks, 128 >>>(in_on_device, out_on_device, size);
    cudaDeviceSynchronize();

    /* Copy result vector back from device to host */
    cudaMemcpy(out.element, out_on_device, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

    /* Free memory on GPU */
    cudaFree(in_on_device);
    cudaFree(out_on_device);

    return;
}

/* Check correctness of results */
int check_results(const float *pix1, const float *pix2, int num_elements, float eps) 
{
    int i;
    for (i = 0; i < num_elements; i++)
        if (fabsf((pix1[i] - pix2[i])/pix1[i]) > eps) 
            return -1;
    
    return 0;
}

/* Print out the image contents */
void print_image(const image_t img)
{
    int i, j;
    float val;
    for (i = 0; i < img.size; i++) {
        for (j = 0; j < img.size; j++) {
            val = img.element[i * img.size + j];
            printf("%0.4f ", val);
        }
        printf("\n");
    }

    printf("\n");
}
