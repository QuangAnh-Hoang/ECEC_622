#include "jacobi_iteration.h"

/* FIXME: Write the device kernels to solve the Jacobi iterations */


__global__ void jacobi_iteration_kernel_naive(const float *A, float *src, float *dest, const float *b, const int size, float *ssd)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int i;
    float sum = 0.0;
    for (i = 0; i < size; i++) {
        if (i != tid) {
            sum += A[tid*size + i] * src[i];
        }
    }

    dest[tid] = (b[tid] - sum)/A[tid*size + tid];

    float local_diff = (dest[tid] - src[tid])*(dest[tid] - src[tid]);
    __syncthreads();

    atomicAdd(ssd, local_diff);

    return;
}

__global__ void jacobi_iteration_kernel_optimized()
{
    return;
}

