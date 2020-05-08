/* Code for the Jacobi method of solving a system of linear equations 
 * by iteration.

 * Author: Naga Kandasamy
 * Date modified: April 22, 2020
 *
 * Compile as follows:
 * gcc -o jacobi_solver jacobi_solver.c compute_gold.c -Wall -O3 -lpthread -lm 
*/

#define _REENTRANT /* Make sure the library functions are MT (muti-thread) safe */
#define _GNU_SOURCE

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <semaphore.h>
#include <pthread.h>
#include "jacobi_solver.h"

/* Uncomment the line below to spit out debug information */ 
/* #define DEBUG */

int main(int argc, char **argv) 
{
	if (argc < 2) {
		fprintf(stderr, "Usage: %s matrix-size\n", argv[0]);
        fprintf(stderr, "matrix-size: width of the square matrix\n");
		exit(EXIT_FAILURE);
	}

    int matrix_size = atoi(argv[1]);
	int num_threads = atoi(argv[2]);

    matrix_t  A;                    /* N x N constant matrix */
	matrix_t  B;                    /* N x 1 b matrix */
	matrix_t reference_x;           /* Reference solution */ 
    matrix_t mt_solution_x;         /* Solution computed by pthread code */

	/* Generate diagonally dominant matrix */
    fprintf(stderr, "\nCreating input matrices\n");
	srand(time(NULL));
	A = create_diagonally_dominant_matrix(matrix_size, matrix_size);
	if (A.elements == NULL) {
        fprintf(stderr, "Error creating matrix\n");
        exit(EXIT_FAILURE);
	}
	
    /* Create other matrices */
    B = allocate_matrix(matrix_size, 1, 1);
	reference_x = allocate_matrix(matrix_size, 1, 0);
	mt_solution_x = allocate_matrix(matrix_size, 1, 0);

#ifdef DEBUG
	print_matrix(A);
	print_matrix(B);
	print_matrix(reference_x);
#endif

    /* Compute Jacobi solution using reference code */
	fprintf(stderr, "Generating solution using reference code\n");
    int max_iter = 100000; /* Maximum number of iterations to run */
	struct timeval start, stop;
    gettimeofday(&start, NULL);
    compute_gold(A, reference_x, B, max_iter);
	gettimeofday(&stop, NULL);
    display_jacobi_solution(A, reference_x, B); /* Display statistics */

    float ref_time = (float)(stop.tv_sec - start.tv_sec \
                + (stop.tv_usec - start.tv_usec) / (float)1000000);
    fprintf(stderr, "CPU run time = %0.6f s\n", ref_time);

	/* Compute the Jacobi solution using pthreads. 
     * Solutions are returned in mt_solution_x.
     * */
    fprintf(stderr, "\nPerforming Jacobi iteration using pthreads\n");
	gettimeofday(&start, NULL);
	compute_using_pthreads(A, mt_solution_x, B, num_threads);
	gettimeofday(&stop, NULL);
    display_jacobi_solution(A, mt_solution_x, B); /* Display statistics */

	float mt_time = (float)(stop.tv_sec - start.tv_sec \
                + (stop.tv_usec - start.tv_usec) / (float)1000000);
    fprintf(stderr, "MT run time = %0.6f s\n", mt_time);

    float speedup = ref_time / mt_time;
    fprintf(stderr, "Speedup = %0.6f times\n", speedup);

	printf("\n%d\t\t%d\t\t%0.6f", matrix_size, num_threads, speedup);

    free(A.elements); 
	free(B.elements); 
	free(reference_x.elements); 
	free(mt_solution_x.elements);
	
    exit(EXIT_SUCCESS);
}

void jacobi_func(void *args) {
	arg_for_thread_t *thread_data = (arg_for_thread_t *)args;

	int i, j;
	int num_rows = thread_data->A.num_rows;
	int num_cols = thread_data->A.num_columns;
	double partial_diff = 0.0;

	for (i = thread_data->tid; i < num_rows; i += thread_data->num_threads) {
		double sum = 0.0;
		for (j = 0; j < num_cols; j++) {
			if (j != i) {
				sum += thread_data->A.elements[i*num_cols + j] * (thread_data->src[j]);
			}
		}
		thread_data->dest[i] = (thread_data->B.elements[i] - sum)/thread_data->A.elements[i*num_cols + i];
		partial_diff += (thread_data->dest[i] - thread_data->src[i]) * \
						(thread_data->dest[i] - thread_data->src[i]);
	}
	pthread_mutex_lock(thread_data->mutex_mse);
	*thread_data->ssd += partial_diff;
	pthread_mutex_unlock(thread_data->mutex_mse);
}

/* FIXME: Complete this function to perform the Jacobi calculation using pthreads. 
 * Result must be placed in mt_sol_x. */
void compute_using_pthreads (const matrix_t A, matrix_t mt_sol_x, const matrix_t B, int num_threads)
{
	int i;
	int num_iter = 0;
	double ssd = 1;

	matrix_t new_x = allocate_matrix(A.num_rows, 1, 0);
	pthread_attr_t attributes;
	pthread_mutex_t mutex_mse;
	pthread_attr_init(&attributes);
	pthread_mutex_init(&mutex_mse, NULL);

	pthread_t *worker = (pthread_t *) malloc (num_threads * sizeof(pthread_t));
	arg_for_thread_t *thread_data;
	while (sqrt(ssd) > THRESHOLD) {
		ssd = 0.0;
		num_iter++;
		for (i = 0; i < num_threads; i++) {
			thread_data = (arg_for_thread_t *) malloc (sizeof(arg_for_thread_t));
			thread_data->tid = i;
			thread_data->num_iter = num_iter;
			thread_data->num_threads = num_threads;
			thread_data->ssd = &ssd;
			thread_data->A = A;
			thread_data->B = B;
			thread_data->mutex_mse = &mutex_mse;
			if ((num_iter % 2) == 0) {
				thread_data->src = mt_sol_x.elements;
				thread_data->dest = new_x.elements;
			}
			else {
				thread_data->dest = mt_sol_x.elements;
				thread_data->src = new_x.elements;
			}
			if ((pthread_create(&worker[i], NULL, jacobi_func, (void *)thread_data)) != 0) {
				perror("pthread_create");
				exit(EXIT_FAILURE);
			}
		}
		for (i = 0; i < num_threads; i++) {
			pthread_join(worker[i], NULL);
		}
		fprintf(stderr, "Iteration: %d. MSE = %f\n", num_iter, sqrt(ssd));
	}
}

/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
*/
matrix_t allocate_matrix(int num_rows, int num_columns, int init)
{
    int i;    
    matrix_t M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
		
	M.elements = (float *)malloc(size * sizeof(float));
	for (i = 0; i < size; i++) {
		if (init == 0) 
            M.elements[i] = 0; 
		else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}
    
    return M;
}	

/* Print matrix to screen */
void print_matrix(const matrix_t M)
{
    int i, j;
	for (i = 0; i < M.num_rows; i++) {
        for (j = 0; j < M.num_columns; j++) {
			fprintf(stderr, "%f ", M.elements[i * M.num_rows + j]);
        }
		
        fprintf(stderr, "\n");
	} 
	
    fprintf(stderr, "\n");
    return;
}

/* Return a floating-point value between [min, max] */
float get_random_number(int min, int max)
{
    float r = rand ()/(float)RAND_MAX;
	return (float)floor((double)(min + (max - min + 1) * r));
}

/* Check if matrix is diagonally dominant */
int check_if_diagonal_dominant(const matrix_t M)
{
    int i, j;
	float diag_element;
	float sum;
	for (i = 0; i < M.num_rows; i++) {
		sum = 0.0; 
		diag_element = M.elements[i * M.num_rows + i];
		for (j = 0; j < M.num_columns; j++) {
			if (i != j)
				sum += abs(M.elements[i * M.num_rows + j]);
		}
		
        if (diag_element <= sum)
			return -1;
	}

	return 0;
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix (int num_rows, int num_columns)
{
	matrix_t M;
	M.num_columns = num_columns;
	M.num_rows = num_rows; 
	int size = M.num_rows * M.num_columns;
	M.elements = (float *)malloc(size * sizeof(float));

    int i, j;
	fprintf(stderr, "Generating %d x %d matrix with numbers between [-.5, .5]\n", num_rows, num_columns);
	for (i = 0; i < size; i++)
        M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	
	/* Make diagonal entries large with respect to the entries on each row. */
    float row_sum;
	for (i = 0; i < num_rows; i++) {
		row_sum = 0.0;		
		for (j = 0; j < num_columns; j++) {
			row_sum += fabs(M.elements[i * M.num_rows + j]);
		}
		
        M.elements[i * M.num_rows + i] = 0.5 + row_sum;
	}

    /* Check if matrix is diagonal dominant */
	if (check_if_diagonal_dominant(M) < 0) {
		free(M.elements);
		M.elements = NULL;
	}
	
    return M;
}



