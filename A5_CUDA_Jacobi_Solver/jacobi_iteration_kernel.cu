#include "jacobi_iteration.h"

/* FIXME: Write the device kernels to solve the Jacobi iterations */


__device__ void lock(int *mutex)
{
    while (atomicCAS(mutex, 0, 1) != 0);
    return;
}

__device__ void unlock(int *mutex)
{
    atomicExch(mutex, 0);
    return;
}

__global__ void jacobi_iteration_kernel_naive(const float *A, float *src, float *dest, const float *b, const int size, float *ssd, int *mutex)
{
    __shared__ float diff_per_row[THREAD_BLOCK_SIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < size) {
        int i;
        float sum = 0.0;
        for (i = 0; i < size; i++) {
            if (i != tid) {
                sum += A[tid*size + i] * src[i];
            }
        }
    
        dest[tid] = (b[tid] - sum)/A[tid*size + tid];
    
        diff_per_row[threadIdx.x] = (dest[tid] - src[tid])*(dest[tid] - src[tid]);
        __syncthreads();

	i = blockDim.x/2;
	while (i != 0) {
	    if (threadIdx.x < i) {
		diff_per_row[threadIdx.x] += diff_per_row[threadIdx.x + i];
	    }
	    __syncthreads();
	    i /= 2;
	}

	if (threadIdx.x == 0) {
	    lock(mutex);
	    *ssd += diff_per_row[0];
	    unlock(mutex);
	}
    }

    return;
}

__global__ void jacobi_iteration_kernel_optimized()
{
    return;
}

