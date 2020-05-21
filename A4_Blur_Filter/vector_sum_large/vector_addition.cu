/* This code illustrates the use of the GPU to perform vector addition on arbirarily large vectors. 
    Author: Naga Kandasamy
    Date modified: May 3, 2020
*/  
  
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* Include kernel code during preprocessing step */
#include "vector_addition_kernel.cu"
  
#define THREAD_BLOCK_SIZE 128
#define NUM_THREAD_BLOCKS 240
  
void run_test(int);
void compute_on_device(float *, float *, float *, int);
void check_for_error(char const *);
extern "C" void compute_gold(float *, float *, float *, int);

int main(int argc, char **argv) 
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s num-elements\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int num_elements = atoi(argv[1]);
    run_test(num_elements);
	
    exit(EXIT_SUCCESS);
}

/* Perform vector addition on CPU and GPU */
void run_test(int num_elements)                
{
    float diff;
    int i;
		 			 
    /* Allocate memory on CPU for input vectors A and B, and output vector C */
     int vector_length = sizeof(float) * num_elements;
     float *A = (float *)malloc(vector_length);
     float *B = (float *)malloc(vector_length);
     float *gold_result = (float *)malloc(vector_length);	/* Result vector computed on CPU */
     float *gpu_result = (float *)malloc(vector_length);	/* Result vector computed on GPU */
			 
     /* Initialize input data to be integer values between 0 and 5 */ 
     for (i = 0; i < num_elements; i++) {
         A[i] = floorf(5 * (rand() / (float)RAND_MAX));
         B[i] = floorf(5 * (rand() / (float)RAND_MAX));
     } 
				
     /* Compute reference solution on CPU */
     fprintf(stderr, "Adding vectors on the CPU\n");
     compute_gold(A, B, gold_result, num_elements);
					 	  
     /* Compute result vector on GPU */ 
     compute_on_device(A, B, gpu_result, num_elements);
  
     /* Compute differences between the CPU and GPU results. */
     diff = 0.0;
     for (i = 0; i < num_elements; i++)
         diff += fabsf(gold_result[i] - gpu_result[i]);
	
     fprintf(stderr, "Difference between the CPU and GPU result: %f\n", diff);
  
     /* Free the data structures. */
     free((void *)A); 
     free((void *)B);
     free((void *)gold_result); 
     free((void *)gpu_result);
	
     exit(EXIT_SUCCESS);
}

/* Host side code. 
   
   Transfer vectors A and B from CPU to GPU, set up grid and 
   thread dimensions, execute kernel function, and copy result vector 
   back to CPU. 
 */
void compute_on_device(float *A_on_host, float *B_on_host, float *gpu_result, int num_elements)
{
    float *A_on_device = NULL;
    float *B_on_device = NULL;
    float *C_on_device = NULL;
	
    /* Allocate space on GPU for vectors A and B, and copy contents of vectors to GPU */
    cudaMalloc((void **)&A_on_device, num_elements * sizeof(float));
    cudaMemcpy(A_on_device, A_on_host, num_elements * sizeof(float), cudaMemcpyHostToDevice);
	
    cudaMalloc((void **)&B_on_device, num_elements * sizeof(float));
    cudaMemcpy(B_on_device, B_on_host, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    
    /* Allocate space for result vector on GPU */
    cudaMalloc((void **)&C_on_device, num_elements * sizeof(float));
  
    /* Set up the execution grid on the GPU. */
    int num_thread_blocks = NUM_THREAD_BLOCKS;
    dim3 thread_block(THREAD_BLOCK_SIZE, 1, 1);	/* Set number of threads in the thread block */
    fprintf(stderr, "Setting up a (%d x 1) execution grid\n", num_thread_blocks);
    dim3 grid(num_thread_blocks, 1);
	
    fprintf(stderr, "Adding vectors on the GPU\n");
  
    /* Launch kernel with multiple thread blocks. The kernel call is non-blocking. */
    vector_addition_kernel<<< grid, thread_block >>>(A_on_device, B_on_device, C_on_device, num_elements);	 
    cudaDeviceSynchronize(); /* Force CPU to wait for GPU to complete */
    check_for_error("KERNEL FAILURE");
  
    /* Copy result vector back from GPU and store */
    cudaMemcpy(gpu_result, C_on_device, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
  
	/* Free memory on GPU */	  
    cudaFree(A_on_device); 
    cudaFree(B_on_device); 
    cudaFree(C_on_device);
} 

/* Check for errors when executing the kernel */
void check_for_error(char const *msg)                
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA ERROR: %s (%s). \n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


