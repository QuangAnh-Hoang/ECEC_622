/* Write GPU code to perform the step(s) involved in counting sort. 
 Add additional kernels and device functions as needed. */

__global__ void counting_sort_kernel(int *input_array, int *sorted_array, int *histogram,\
    int *prefix_array, int num_elements, int range, int *global_mutex)
{
    __shared__ int input_s[THREAD_BLOCK_SIZE];
    __shared__ int hist_s[MAX_VALUE - MIN_VALUE + 1];
    __shared__ int mutex = 0;

    int i, start_idx, stop_idx;
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int num_bins = MAX_VALUE - MIN_VALUE + 1;

    /* Histogram generation */
    if (tid < num_elements) {
        input_s[threadIdx.x] = input_array[blockIdx.x*blockDim.x + threadIdx.x];
        
        i = input_s[threadIdx.x];
        while (atomicCAS(mutex, 0, 1) != 0);
        hist_s[i]++;
        atomicExch(mutex, 0);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        while (atomicCAS(global_mutex, 0, 1) != 0);
        for (i = 0; i < num_bins, i++) {
            histogram[i] += hist_s[i];
        }
        atomicExch(global_mutex, 0);
    }
    __syncthreads();

    /* Inclusive prefix scan */
    if (tid < num_bins) {
        for (i = 0; i <= tid; i++) {
            prefix_array[tid] += hist_s[i];
        }
    }
    __syncthreads();

    /* Generate sorted array */
    if (tid < num_bins) {
        start_idx = 0;
        stop_idx = prefix_array[tid];
        if (tid > 0) {
            start_idx = prefix_array[tid - 1];
        }
        for (i = start_idx; i < stop_idx; i++) {
            sorted_array[i] = tid;
        }
    }
    __syncthreads();

    return;
}
