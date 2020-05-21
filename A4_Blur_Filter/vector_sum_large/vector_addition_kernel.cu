#ifndef _VECTOR_ADDITION_KERNEL_H_
#define _VECTOR_ADDITION_KERNEL_H_

__global__ void vector_addition_kernel(float *A, float *B, float *C, int num_elements)
{
    /* Obtain index of thread within the overall execution grid */
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x; 
    /* Compute the stride length = total number of threads */
    int stride = blockDim.x * gridDim.x; 		  
    
    while (thread_id < num_elements) {
        C[thread_id] = A[thread_id] + B[thread_id];
        thread_id += stride;
    }
		  
    return; 
}

#endif /* #ifndef _VECTOR_ADDITION_KERNEL_H_ */
