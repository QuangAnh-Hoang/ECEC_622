#include "jacobi_iteration.h"

/* Write the device kernels to solve the Jacobi iterations */


__global__ void jacobi_iteration_kernel_naive(const matrix_t A, matrix_t src, matrix_t dest,
    const matrix_t b, const int size, int *mutex, matrix_t ssd)
{
    __shared__ float sum_per_thread[THREAD_BLOCK_SIZE];
    int tid = threadIdx.x + blockIdx.x*blockDim.x;

    if (tid < size) {
        int i;
        float sum = 0;
        for (i = 0; i < size; i++) {
            if (i != tid) {
                sum += A.elements[tid*size + i]*src.elements[i];
            }
        }
        dest.elements[tid] = (b.elements[tid] - sum)/A.elements[tid*size + tid];
        sum_per_thread[threadIdx.x] = (dest.elements[tid] - src.elements[tid])*(dest.elements[tid] - src.elements[tid]);
        __syncthreads();

        i = blockDim.x / 2;
        while (i != 0) {
            if (threadIdx.x < i) {
                sum_per_thread[threadIdx.x] += sum_per_thread[threadIdx.x + i];
            }
            __syncthreads();
            i /= 2;
        }

        if (threadIdx.x == 0) {
            while(atomicCAS(mutex, 0, 1) != 0);
            ssd.elements[0] += sum_per_thread[0];
            atomicExch(mutex, 0);
        }
    }

    return;
}

__global__ void jacobi_iteration_kernel_optimized(const matrix_t col_A, matrix_t src, matrix_t dest, \
    const matrix_t b, const int size, int *mutex, matrix_t ssd)
{
    __shared__ float A_tile[THREAD_BLOCK_SIZE];
    __shared__ float b_tile[THREAD_BLOCK_SIZE];
    __shared__ float curr_x;
    __shared__ float sum_per_thread[THREAD_BLOCK_SIZE];

    int tid = threadIdx.x + blockIdx.x*blockDim.x;

    if (tid < size) {
        int i;
        float sum = 0;
        for (i = 0; i < size; i++) {
            A_tile[threadIdx.x] = col_A.elements[tid + i*size];
            curr_x = src.elements[i];
            if (i != tid) {
                sum += A_tile[threadIdx.x]*curr_x;
            }
        }
        b_tile[threadIdx.x] = b.elements[tid];
        dest.elements[tid] = (b_tile[threadIdx.x] - sum)/col_A.elements[tid*size + tid];
        sum_per_thread[threadIdx.x] = (dest.elements[tid] - src.elements[tid])*(dest.elements[tid] - src.elements[tid]);
        __syncthreads();

        i = blockDim.x / 2;
        while (i != 0) {
            if (threadIdx.x < i) {
                sum_per_thread[threadIdx.x] += sum_per_thread[threadIdx.x + i];
            }
            __syncthreads();
            i /= 2;
        }

        if (threadIdx.x == 0) {
            while(atomicCAS(mutex, 0, 1) != 0);
            ssd.elements[0] += sum_per_thread[0];
            atomicExch(mutex, 0);
        }
    }

    return;
}

