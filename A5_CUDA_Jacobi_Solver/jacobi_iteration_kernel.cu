#include "jacobi_iteration.h"

/* FIXME: Write the device kernels to solve the Jacobi iterations */


__global__ void jacobi_iteration_kernel_naive(const double *A, const double *src, double *dest, 
    const double *b, const int size, int *mutex, double *ssd)
{
    __shared__ double sum_per_thread[THREAD_BLOCK_SIZE];

    int tid = blockDim.x*blockIdx.x + threadIdx.x;

    if (tid < size) {
        int i;
        double sum = 0;
        double diff = 0;
        for (i = 0; i < size; i++) {
            if (i != tid) {
                sum += A[tid*size + i]*src[i];
            }
        }
        dest[tid] = (b[tid] - sum)/A[tid*size + tid];
        diff = dest[tid] - src[tid];
        sum_per_thread[threadIdx.x] = diff*diff;
        __syncthreads();

        i = blockDim.x/2;
        while (i != 0) {
            if (threadIdx.x < i) {
                sum_per_thread[threadIdx.x] += sum_per_thread[threadIdx.x + i];
            }
            __syncthreads();
            i /= 2;
        }

        if (threadIdx.x == 0) {
            while (atomicCAS(mutex, 0, 1) != 0);
            *ssd += (double) sum_per_thread[0];
            atomicExch(mutex, 0);
        }
    }
    return;
}

__global__ void jacobi_iteration_kernel_optimized(const double *col_A, const double *src, double *dest, 
    const double *b, const int size, int *mutex, double *ssd)
{
    __shared__ double A_tile[THREAD_BLOCK_SIZE];
    __shared__ double b_tile[THREAD_BLOCK_SIZE];
    __shared__ double curr_x;
    __shared__ double sum_per_thread[THREAD_BLOCK_SIZE];

    int tid = threadIdx.x + blockIdx.x*blockDim.x;

    if (tid < size) {
        int i;
        double sum = 0;
        double diff = 0;
        for (i = 0; i < size; i++) {
            A_tile[threadIdx.x] = col_A[tid + i*size];
            curr_x = src[i];
            if (i != tid) {
                sum += A_tile[threadIdx.x]*curr_x;
            }
        }
        b_tile[threadIdx.x] = b[tid];
        dest[tid] = (b_tile[threadIdx.x] - sum)/col_A[tid*size + tid];
        diff = dest[tid] - src[tid];
        sum_per_thread[threadIdx.x] = diff*diff;
        __syncthreads();

        i = blockDim.x/2;
        while (i != 0) {
            if (threadIdx.x < i) {
                sum_per_thread[threadIdx.x] += sum_per_thread[threadIdx.x + i];
            }
            __syncthreads();
            i /= 2;
        }

        if (threadIdx.x == 0) {
            while (atomicCAS(mutex, 0, 1) != 0);
            *ssd += (double) sum_per_thread[0];
            atomicExch(mutex, 0);
        }
    }

    return;
}

