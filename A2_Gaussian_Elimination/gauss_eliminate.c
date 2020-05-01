/* Gaussian elimination code.
 * 
 * Author: Naga Kandasamy
 * Date of last update: April 22, 2020
 *
 * Student names(s): FIXME
 * Date: FIXME
 *
 * Compile as follows: 
 * gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -O3 -Wall -lpthread -lm
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "gauss_eliminate.h"

#define MIN_NUMBER 2
#define MAX_NUMBER 50

/* Function prototypes */
extern int compute_gold(float *, int);
Matrix allocate_matrix(int, int, int);
void gauss_eliminate_using_pthreads(Matrix, unsigned int);
int perform_simple_check(const Matrix);
void print_matrix(const Matrix);
float get_random_number(int, int);
int check_results(float *, float *, int, float);


int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s matrix-size\n", argv[0]);
        fprintf(stderr, "matrix-size: width and height of the square matrix\n");
        exit(EXIT_FAILURE);
    }

    int matrix_size = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    Matrix A;			                                            /* Input matrix */
    Matrix U_reference;		                                        /* Upper triangular matrix computed by reference code */
    Matrix U_mt;			                                        /* Upper triangular matrix computed by pthreads */

    fprintf(stderr, "Generating input matrices\n");
    srand(time (NULL));                                             /* Seed random number generator */
    A = allocate_matrix(matrix_size, matrix_size, 1);               /* Allocate and populate random square matrix */
    U_reference = allocate_matrix (matrix_size, matrix_size, 0);    /* Allocate space for reference result */
    U_mt = allocate_matrix (matrix_size, matrix_size, 0);           /* Allocate space for multi-threaded result */

    /* Copy contents A matrix into U matrices */
    int i, j;
    for (i = 0; i < A.num_rows; i++) {
        for (j = 0; j < A.num_rows; j++) {
            U_reference.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
            U_mt.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
        }
    }

    fprintf(stderr, "\nPerforming gaussian elimination using reference code\n");
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    
    int status = compute_gold(U_reference.elements, A.num_rows);
  
    gettimeofday(&stop, NULL);
    fprintf(stderr, "CPU run time = %0.2f s\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec) / (float)1000000));

    if (status < 0) {
        fprintf(stderr, "Failed to convert given matrix to upper triangular. Try again.\n");
        exit(EXIT_FAILURE);
    }
  
    status = perform_simple_check(U_reference);	/* Check that principal diagonal elements are 1 */ 
    if (status < 0) {
        fprintf(stderr, "Upper triangular matrix is incorrect. Exiting.\n");
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "Single-threaded Gaussian elimination was successful.\n");
  
    /* FIXME: Perform Gaussian elimination using pthreads. 
     * The resulting upper triangular matrix should be returned in U_mt */
    fprintf(stderr, "\nPerforming gaussian elimination using pthreads\n");
    gauss_eliminate_using_pthreads(U_mt, num_threads);

    /* Check if pthread result matches reference solution within specified tolerance */
    fprintf(stderr, "\nChecking results\n");
    int size = matrix_size * matrix_size;
    int res = check_results(U_reference.elements, U_mt.elements, size, 1e-6);

    fprintf(stderr, "\nReference: \n");
    print_matrix(U_reference);

    fprintf(stderr, "TEST %s\n", (0 == res) ? "PASSED" : "FAILED");

    /* Free memory allocated for matrices */
    free(A.elements);
    free(U_reference.elements);
    free(U_mt.elements);

    exit(EXIT_SUCCESS);
}

void divide(void *args) {
    args_for_div_t *thread_data = (args_for_div_t *) args;
    int i;
    int offset = thread_data->tid*thread_data->chunksize;
    int endpoint;
    if (thread_data->tid < (thread_data->num_threads - 1)) {
        for (i = offset; i < offset + thread_data->chunksize; i++) {
            if (i < thread_data->current_row) {
                thread_data->U[thread_data->current_row*thread_data->dim + i] = 0;
            }
            else {
                thread_data->U[thread_data->current_row*thread_data->dim + i] = (float)\
                    thread_data->U[thread_data->current_row*thread_data->dim + i]/thread_data->root_value;                
            }
        }
    }
    else {
        for (i = offset; i < thread_data->dim; i++) {
            if (i < thread_data->current_row) {
                thread_data->U[thread_data->current_row*thread_data->dim + i] = 0;
            }
            else {
                thread_data->U[thread_data->current_row*thread_data->dim + i] = (float)\
                    thread_data->U[thread_data->current_row*thread_data->dim + i]/thread_data->root_value;                
            }
        }
    }
    free(thread_data);
}

void eliminate(void *args) {
    args_for_eli_t *thread_data = (args_for_eli_t *) args;
    int i, j, k;
    k = thread_data->current_row;
    i = k + 1 + thread_data->tid;
    int stride = thread_data->num_threads;
    while (i < thread_data->dim) {
        float root_val = thread_data->U[thread_data->dim*i + k];
        for (j = k; j < thread_data->dim; j++) {
            thread_data->U[thread_data->dim*i + j] -= root_val*thread_data->U[thread_data->dim*k + j];
        }
        i += stride;
    }
    free(thread_data);
}

/* FIXME: Write code to perform gaussian elimination using pthreads */
void gauss_eliminate_using_pthreads(Matrix U, unsigned int num_threads)
{
    int i, j;
    for (i = 0; i < U.num_rows; i++) {
        pthread_t *worker = (pthread_t *) malloc (num_threads * sizeof(pthread_t));
        args_for_div_t *div_data;
        args_for_eli_t *eli_data;
        float root_value = U.elements[i*U.num_columns + i];
        for (j = 0; j < num_threads; j++) {
            div_data = (args_for_div_t *) malloc (sizeof(args_for_div_t));
            div_data->tid = j;
            div_data->current_row = i;
            div_data->dim = U.num_columns;
            div_data->root_value = root_value;
            int chunksize = (int) floor(U.num_columns/num_threads);
            if (chunksize == 0) {
                div_data->chunksize = 1;
            }
            else {
                div_data->chunksize = chunksize;
            }
            div_data->U = U.elements;

            if ((pthread_create(&worker[j], NULL, divide, (void *) div_data)) != 0) {
                perror("pthread_create_divide");
                exit(EXIT_FAILURE);
            }
        }

        for (j = 0; j < num_threads; j++) {
            pthread_join(worker[j], NULL);
        }

        fprintf(stderr, "\nDivision: ");
        print_matrix(U);

        for (j = 0; j < num_threads; j++) {
            eli_data = (args_for_eli_t *) malloc (sizeof(args_for_eli_t));
            eli_data->tid = j;
            eli_data->current_row = i;
            eli_data->dim = U.num_columns;
            eli_data->num_threads = num_threads;
            eli_data->U = U.elements;

            if ((pthread_create(&worker[j], NULL, eliminate, (void *) eli_data)) != 0) {
                perror("pthread_create_eliminate");
                exit(EXIT_FAILURE);
            }
        }

        for (j = 0; j < num_threads; j++) {
            pthread_join(worker[j], NULL);
        }

        fprintf(stderr, "\nElimination: ");
        print_matrix(U);

        fprintf(stderr, "\n----------------------------------------\n");
    }
    
}


/* Check if results generated by single threaded and multi threaded versions match within tolerance */
int check_results(float *A, float *B, int size, float tolerance)
{
    int i;
    for (i = 0; i < size; i++)
        if(fabsf(A[i] - B[i]) > tolerance)
            return -1;
    return 0;
}


/* Allocate a matrix of dimensions height*width
 * If init == 0, initialize to all zeroes.  
 * If init == 1, perform random initialization. 
*/
Matrix allocate_matrix(int num_rows, int num_columns, int init)
{
    int i;
    Matrix M;
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

/* Return a random floating-point number between [min, max] */ 
float get_random_number(int min, int max)
{
    return (float)floor((double)(min + (max - min + 1) * ((float)rand() / (float)RAND_MAX)));
}

/* Perform simple check on upper triangular matrix if the principal diagonal elements are 1 */
int perform_simple_check(const Matrix M)
{
    int i;
    for (i = 0; i < M.num_rows; i++)
        if ((fabs(M.elements[M.num_rows * i + i] - 1.0)) > 1e-6)
            return -1;
  
    return 0;
}

void print_matrix(Matrix A) {
    int i, j;
    int dim = A.num_columns;
    for (i = 0; i < dim; i++) {
        fprintf(stderr, "\n");
        for (j = 0; j < dim; j++) {
            fprintf(stderr, "%.2f, ", A.elements[dim*i + j]);
        }
    }
}
