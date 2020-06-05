/* Host code for the Jacobi method of solving a system of linear equations 
 * by iteration.

 * Build as follws: make clean && make

 * Author: Naga Kandasamy
 * Date modified: May 21, 2020
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "jacobi_iteration.h"

/* Include the kernel code */
#include "jacobi_iteration_kernel.cu"

/* Uncomment the line below if you want the code to spit out debug information. */ 
/* #define DEBUG */

int main(int argc, char **argv) 
{
	if (argc > 1) {
		fprintf(stderr,"This program accepts no arguments\n");
		exit(EXIT_FAILURE);
	}

    matrix_t  A;                    /* N x N constant matrix */
	matrix_t  B;                    /* N x 1 b matrix */
	matrix_t reference_x;           /* Reference solution */ 
	matrix_t gpu_naive_solution_x;  /* Solution computed by naive kernel */
    matrix_t gpu_opt_solution_x;    /* Solution computed by optimized kernel */

    struct timeval start, stop;

	/* Initialize the random number generator */
	srand(time(NULL));

	/* Generate diagonally dominant matrix */ 
    fprintf(stderr, "\nGenerating %d x %d system\n", MATRIX_SIZE, MATRIX_SIZE);
	A = create_diagonally_dominant_matrix(MATRIX_SIZE, MATRIX_SIZE);
	if (A.elements == NULL) {
        fprintf(stderr, "Error creating matrix\n");
        exit(EXIT_FAILURE);
	}
	
    /* Create the other vectors */
    B = allocate_matrix_on_host(MATRIX_SIZE, 1, 1);
	reference_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);
	gpu_naive_solution_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);
    gpu_opt_solution_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);

#ifdef DEBUG
	print_matrix(A);
	print_matrix(B);
	print_matrix(reference_x);
#endif

    /* Compute Jacobi solution on CPU */
    fprintf(stderr, "\nPerforming Jacobi iteration on the CPU\n");
    gettimeofday(&start, NULL);

    compute_gold(A, reference_x, B);

    gettimeofday(&stop, NULL);
    display_jacobi_solution(A, reference_x, B); /* Display statistics */
    double ref_time = (double)(stop.tv_sec - start.tv_sec + \
        (stop.tv_usec - start.tv_usec)/(double)1000000);
    
	/* Compute Jacobi solution on device. Solutions are returned 
       in gpu_naive_solution_x and gpu_opt_solution_x. */
    fprintf(stderr, "\nPerforming Jacobi iteration on device\n");
	compute_on_device(A, gpu_naive_solution_x, gpu_opt_solution_x, B, ref_time);
    display_jacobi_solution(A, gpu_naive_solution_x, B); /* Display statistics */
    display_jacobi_solution(A, gpu_opt_solution_x, B); 
    
    free(A.elements); 
	free(B.elements); 
	free(reference_x.elements); 
	free(gpu_naive_solution_x.elements);
    free(gpu_opt_solution_x.elements);
	
    exit(EXIT_SUCCESS);
}


/* FIXME: Complete this function to perform Jacobi calculation on device */
void compute_on_device(const matrix_t A, matrix_t gpu_naive_sol_x, 
                       matrix_t gpu_opt_sol_x, const matrix_t B, double ref_time)
{
    struct timeval start, stop;

    gettimeofday(&start, NULL);

    /* Allocate matrices on device and transfer data from host to device */
    double *d_A = NULL;
    cudaMalloc((void **)&d_A, (MATRIX_SIZE*MATRIX_SIZE*sizeof(double)));
    check_CUDA_error("Allocate matrix A on device");
    cudaMemcpy(d_A, A.elements, (MATRIX_SIZE*MATRIX_SIZE*sizeof(double)), cudaMemcpyHostToDevice);
    check_CUDA_error("Transfer matrix A's elements from host to device");

    double *d_b = NULL;
    cudaMalloc((void **)&d_b, (MATRIX_SIZE*sizeof(double)));
    check_CUDA_error("Allocate matrix b on device");
    cudaMemcpy(d_b, B.elements, (MATRIX_SIZE*sizeof(double)), cudaMemcpyHostToDevice);
    check_CUDA_error("Transfer matrix b's elements from host to device");

    double *d_x_now = NULL;
    cudaMalloc((void **)&d_x_now, (MATRIX_SIZE*sizeof(double)));
    check_CUDA_error("Allocate matrix x_now on device");
    cudaMemcpy(d_x_now, B.elements, (MATRIX_SIZE*sizeof(double)), cudaMemcpyHostToDevice);
    check_CUDA_error("Initialize matrix x_now's elements on device");

    double *d_x_next = NULL;
    cudaMalloc((void **)&d_x_next, (MATRIX_SIZE*sizeof(double)));
    check_CUDA_error("Allocate matrix x_next on device");
    cudaMemcpy(d_x_next, B.elements, (MATRIX_SIZE*sizeof(double)), cudaMemcpyHostToDevice);
    check_CUDA_error("Initialize matrix x_next's elements on device");

    double *d_ssd = NULL;
    cudaMalloc((void **)&d_ssd, sizeof(double));
    check_CUDA_error("Allocate pointer to SSD on device");
    cudaMemset(d_ssd, 0, sizeof(double));
    check_CUDA_error("Initialize SSD's value on device");

    int *mutex_on_device = NULL;
	cudaMalloc((void **)&mutex_on_device, sizeof(int));
    cudaMemset(mutex_on_device, 0, sizeof(int));
    check_CUDA_error("allocate mutex on device");

    /* Set up execution grid */
    dim3 threadPerBlock(THREAD_BLOCK_SIZE, 1);
    dim3 numBlocks((MATRIX_SIZE + THREAD_BLOCK_SIZE - 1)/THREAD_BLOCK_SIZE, 1);

    /* Launch kernel */
    double mse, ssd;
    int num_iter = 0;
    int done = 0;

    while (!done) {
        ssd = 0;
        cudaMemset(d_ssd, 0, sizeof(double));
        check_CUDA_error("Reset SSD to zero on device");

        if (num_iter % 2) {
            jacobi_iteration_kernel_naive<<< numBlocks, threadPerBlock >>>\
                (d_A, d_x_next, d_x_now, d_b, MATRIX_SIZE, mutex_on_device, d_ssd);
        }
        else {
            jacobi_iteration_kernel_naive<<< numBlocks, threadPerBlock >>>\
                (d_A, d_x_now, d_x_next, d_b, MATRIX_SIZE, mutex_on_device, d_ssd);
        }
        cudaError_t dev_sync = cudaDeviceSynchronize();
        if (dev_sync != cudaSuccess) {
            fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(dev_sync),\
                __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }

        cudaMemcpy(&ssd, d_ssd, sizeof(double), cudaMemcpyDeviceToHost);
        check_CUDA_error("Retrieve SSD value from device");

        num_iter++;
        mse = sqrt(ssd);
        fprintf(stderr, "[Naive] Iteration: %d. MSE = %0.8f\n", num_iter, mse);

        if (mse <= THRESHOLD) {
            fprintf(stderr, "The function has converged\n");
            done = 1;
        }
    }

    if (num_iter % 2) {
        cudaMemcpy(gpu_naive_sol_x.elements, d_x_next, (MATRIX_SIZE*sizeof(double)), cudaMemcpyDeviceToHost);
    }
    else {
        cudaMemcpy(gpu_naive_sol_x.elements, d_x_now, (MATRIX_SIZE*sizeof(double)), cudaMemcpyDeviceToHost);
    }

    gettimeofday(&stop, NULL);
    double naive_time = (double)(stop.tv_sec - start.tv_sec + \
        (stop.tv_usec - start.tv_usec)/(double)1000000);

    /*---------------------------------------------------*/
    /*/ / / / / / / / / / / / / / / / / / / / / / / / / /*/
    /*---------------------------------------------------*/

    gettimeofday(&start, NULL);

    matrix_t col_A = convert_to_col_major(A);
    if (col_A.elements == NULL) {
        fprintf(stderr, "[Opt] Fail to convert matrix A to column major\n");
        exit(EXIT_FAILURE);
    }
    double *d_col_A = NULL;
    cudaMalloc((void **)&d_col_A, (MATRIX_SIZE*MATRIX_SIZE*sizeof(double)));
    check_CUDA_error("Allocate column-major matrix A on device");
    cudaMemcpy(d_col_A, col_A.elements, (MATRIX_SIZE*MATRIX_SIZE*sizeof(double)), cudaMemcpyHostToDevice);
    check_CUDA_error("Transfer column-major matrix A's elements from host to device");

    cudaMemcpy(d_x_now, B.elements, (MATRIX_SIZE*sizeof(double)), cudaMemcpyHostToDevice);
    check_CUDA_error("Reset matrix x_now's elements on device");

    cudaMemcpy(d_x_next, B.elements, (MATRIX_SIZE*sizeof(double)), cudaMemcpyHostToDevice);
    check_CUDA_error("Reset matrix x_next's elements on device");

    /* Launch kernel */
    num_iter = 0;
    done = 0;

    while (!done) {
        ssd = 0;
        cudaMemset(d_ssd, 0, sizeof(double));
        check_CUDA_error("Reset SSD to zero on device");

        if (num_iter % 2) {
            jacobi_iteration_kernel_optimized<<< numBlocks, threadPerBlock >>>\
                (d_A, d_x_next, d_x_now, d_b, MATRIX_SIZE, mutex_on_device, d_ssd);
        }
        else {
            jacobi_iteration_kernel_optimized<<< numBlocks, threadPerBlock >>>\
                (d_A, d_x_now, d_x_next, d_b, MATRIX_SIZE, mutex_on_device, d_ssd);
        }
        cudaError_t dev_sync = cudaDeviceSynchronize();
        if (dev_sync != cudaSuccess) {
            fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(dev_sync),\
                __FILE__, __LINE__);
            exit(EXIT_FAILURE);
        }

        cudaMemcpy(&ssd, d_ssd, sizeof(double), cudaMemcpyDeviceToHost);
        check_CUDA_error("Retrieve SSD value from device");

        num_iter++;
        mse = sqrt(ssd);
        fprintf(stderr, "[Opt] Iteration: %d. MSE = %0.8f\n", num_iter, mse);

        if (mse <= THRESHOLD) {
            fprintf(stderr, "The function has converged\n");
            done = 1;
        }
    }

    if (num_iter % 2) {
        cudaMemcpy(gpu_opt_sol_x.elements, d_x_next, (MATRIX_SIZE*sizeof(double)), cudaMemcpyDeviceToHost);
    }
    else {
        cudaMemcpy(gpu_opt_sol_x.elements, d_x_now, (MATRIX_SIZE*sizeof(double)), cudaMemcpyDeviceToHost);
    }

    gettimeofday(&stop, NULL);
    double opt_time = (double)(stop.tv_sec - start.tv_sec + \
        (stop.tv_usec - start.tv_usec)/(double)1000000);

    fprintf(stderr, "Naive implementation speedup = %0.6f\n", (ref_time/naive_time));
    fprintf(stderr, "Optimized implementation speedup = %0.6f\n", (ref_time/opt_time));

    printf("%4d\t%4d\t%2.6f\t%2.6f\n", \
        MATRIX_SIZE, THREAD_BLOCK_SIZE, (ref_time/naive_time), (ref_time/opt_time));

    cudaFree(&d_A);
    cudaFree(&d_col_A);
    cudaFree(&d_b);
    cudaFree(&d_x_now);
    cudaFree(&d_x_next);
    cudaFree(&d_ssd);

    return;
}

/* Allocate matrix on the device of same size as M */
matrix_t allocate_matrix_on_device(const matrix_t M)
{
    matrix_t Mdevice = M;
    int size = M.num_rows * M.num_columns * sizeof(double);
    cudaMalloc((void **)&Mdevice.elements, size);
    return Mdevice;
}

/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
*/
matrix_t allocate_matrix_on_host(int num_rows, int num_columns, int init)
{	
    matrix_t M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
		
	M.elements = (double *)malloc(size * sizeof(double));
	for (unsigned int i = 0; i < size; i++) {
		if (init == 0) 
            M.elements[i] = 0; 
		else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}
    
    return M;
}	

/* Copy matrix to device */
void copy_matrix_to_device(matrix_t Mdevice, const matrix_t Mhost)
{
    int size = Mhost.num_rows * Mhost.num_columns * sizeof(double);
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
    return;
}

/* Copy matrix from device to host */
void copy_matrix_from_device(matrix_t Mhost, const matrix_t Mdevice)
{
    int size = Mdevice.num_rows * Mdevice.num_columns * sizeof(double);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
    return;
}

/* Prints the matrix out to screen */
void print_matrix(const matrix_t M)
{
	for (unsigned int i = 0; i < M.num_rows; i++) {
        for (unsigned int j = 0; j < M.num_columns; j++) {
			fprintf(stderr, "%f ", M.elements[i * M.num_rows + j]);
        }
		
        fprintf(stderr, "\n");
	} 
	
    fprintf(stderr, "\n");
    return;
}

/* Returns a doubleing-point value between [min, max] */
double get_random_number(int min, int max)
{
    double r = rand()/(double)RAND_MAX;
	return (double)floor((double)(min + (max - min + 1) * r));
}

/* Check for errors in kernel execution */
void check_CUDA_error(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if ( cudaSuccess != err) {
		fprintf(stderr, "CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}	
    
    return;    
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix(unsigned int num_rows, unsigned int num_columns)
{
	matrix_t M;
	M.num_columns = num_columns;
	M.num_rows = num_rows; 
	unsigned int size = M.num_rows * M.num_columns;
	M.elements = (double *)malloc(size * sizeof(double));
    if (M.elements == NULL)
        return M;

	/* Create a matrix with random numbers between [-.5 and .5] */
    unsigned int i, j;
	for (i = 0; i < size; i++)
        M.elements[i] = get_random_number (MIN_NUMBER, MAX_NUMBER);
	
	/* Make diagonal entries large with respect to the entries on each row. */
	for (i = 0; i < num_rows; i++) {
		double row_sum = 0.0;		
		for (j = 0; j < num_columns; j++) {
			row_sum += fabs(M.elements[i * M.num_rows + j]);
		}
		
        M.elements[i * M.num_rows + i] = 0.5 + row_sum;
	}

    return M;
}

/* Convert a row-major matrix to column-major */
matrix_t convert_to_col_major(matrix_t M) {
    matrix_t N;
    N.num_columns = M.num_rows;
    N.num_rows = M.num_columns;
    unsigned int size = N.num_columns * N.num_rows;
    N.elements = (double *)malloc(size * sizeof(double));

    if (N.elements == NULL) {
        return N;
    }

    int i, j;

    for (i = 0; i < N.num_rows; i++) {
        for (j = 0; j < N.num_columns; j++) {
            N.elements[i*N.num_columns + j] = M.elements[j*M.num_columns + i];
        }
    }

    return N;
}