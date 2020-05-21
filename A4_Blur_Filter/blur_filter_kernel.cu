/* Blur filter. Device code. */

#ifndef _BLUR_FILTER_KERNEL_H_
#define _BLUR_FILTER_KERNEL_H_

#include "blur_filter.h"

__global__ void 
blur_filter_kernel (const float *in, float *out, int size)
{
    /* Obtain thread ID */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    /* Compute the stride length = total number of threads */
    int stride = blockDim.x * gridDim.x;

    int i, j;
    int curr_row, curr_col;

    int row = tid/size;
    int col = tid % size;

    float blur_value = 0.0;
    int num_neighbors = 0;
    for (i = -BLUR_SIZE; i < (BLUR_SIZE + 1); i++) {
        for (j = -BLUR_SIZE; j < (BLUR_SIZE + 1); j++) {
            curr_row = row + i;
            curr_col = col + j;
            if ((curr_row > -1) && (curr_row < size) &&\
                (curr_col > -1) && (curr_col < size)) {
                    blur_value += in[curr_row*size + curr_col];
                    num_neighbors++;
                }
        }
    }
    out[tid] = blur_value/num_neighbors;
}

#endif /* _BLUR_FILTER_KERNEL_H_ */
