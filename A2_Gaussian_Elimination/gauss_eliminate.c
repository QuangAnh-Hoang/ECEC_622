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
#include <semaphore.h>
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
void barrier_sync(barrier_t *, int, int);

barrier_t root_val_barrier;
barrier_t divide_barrier;
barrier_t eliminate_barrier;

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s matrix-size\n", argv[0]);
        fprintf(stderr, "matrix-size: width and height of the square matrix\n");
        exit(EXIT_FAILURE);
    }

    int matrix_size = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    /* Initialize the barrier data structure */
    root_val_barrier.counter = 0;
    sem_init(&root_val_barrier.counter_sem, 0, 1); /* Initialize semaphore protecting the counter to unlocked */
    sem_init(&root_val_barrier.barrier_sem, 0, 0); /* Initialize semaphore protecting the barrier to locked */

    /* Initialize the barrier data structure */
    divide_barrier.counter = 0;
    sem_init(&divide_barrier.counter_sem, 0, 1); /* Initialize semaphore protecting the counter to unlocked */
    sem_init(&divide_barrier.barrier_sem, 0, 0); /* Initialize semaphore protecting the barrier to locked */

    /* Initialize the barrier data structure */
    eliminate_barrier.counter = 0;
    sem_init(&eliminate_barrier.counter_sem, 0, 1); /* Initialize semaphore protecting the counter to unlocked */
    sem_init(&eliminate_barrier.barrier_sem, 0, 0); /* Initialize semaphore protecting the barrier to locked */

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
    float ref_time = (float)(stop.tv_sec - start.tv_sec \
                + (stop.tv_usec - start.tv_usec) / (float)1000000);
    fprintf(stderr, "CPU run time = %0.6f s\n", ref_time);

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

    gettimeofday(&start, NULL);

    gauss_eliminate_using_pthreads(U_mt, num_threads);
    
    gettimeofday(&stop, NULL);

    float mt_time = (float)(stop.tv_sec - start.tv_sec \
                + (stop.tv_usec - start.tv_usec) / (float)1000000);
    fprintf(stderr, "MT run time = %0.6f s\n", mt_time);

    float speedup = ref_time / mt_time;

    fprintf(stderr, "Speedup = %0.6f times\n", speedup);

    /* Check if pthread result matches reference solution within specified tolerance */
    fprintf(stderr, "\nChecking results\n");
    int size = matrix_size * matrix_size;
    int res = check_results(U_reference.elements, U_mt.elements, size, 1e-6);
    fprintf(stderr, "TEST %s\n", (0 == res) ? "PASSED" : "FAILED");

    printf("\n%d\t%d\t%0.6f", matrix_size, num_threads, speedup);

    /* Free memory allocated for matrices */
    free(A.elements);
    free(U_reference.elements);
    free(U_mt.elements);

    exit(EXIT_SUCCESS);
}

void divide(int tid, int num_threads, int current_row, float root_value, Matrix *A) {
    int i;
    int chunksize;
    int remaining_elements = A->num_columns - current_row;
    if (remaining_elements < num_threads) {
        if (tid < remaining_elements) {
            chunksize = 1;
        }
        else {
            chunksize = 0;
        }
    }
    else {
        chunksize = (int) floor (remaining_elements/num_threads);        
    }
    int offset = current_row + tid*chunksize;
    if (chunksize > 0) {
        if (tid < num_threads-1) {
            for (i = offset; i < offset + chunksize; i++) {
                A->elements[current_row*A->num_columns + i] = (float)\
                    A->elements[current_row*A->num_columns + i]/root_value;
            }
        }
        else {
            for (i = offset; i < A->num_columns; i++) {
                A->elements[current_row*A->num_columns + i] = (float)\
                    A->elements[current_row*A->num_columns + i]/root_value;
            }
        }
    }
}

void eliminate(int tid, int num_threads, int current_row, Matrix *A) {
    int i, j;
    int chunksize;
    int remain_rows = A->num_rows - (current_row + 1);
    if (remain_rows < num_threads) {
        if (tid < remain_rows) {
            chunksize = 1;
        }
        else {
            chunksize = 0;
        }
    }
    else {
        chunksize = (int) floor (remain_rows/num_threads);        
    }
    int offset = (current_row + 1) + tid*chunksize;
    if (tid < num_threads-1) {
        for (i = offset; i < offset + chunksize; i++) {
            float base_multiplier = A->elements[A->num_columns*i + current_row];
            for (j = current_row; j < A->num_columns; j++) {
                A->elements[A->num_columns*i + j] -= \
                    base_multiplier*A->elements[A->num_columns*current_row + j];
            }
        }
    }
    else {
        for (i = offset; i < A->num_rows; i++) {
            float base_multiplier = A->elements[A->num_columns*i + current_row];
            for (j = current_row; j < A->num_columns; j++) {
                A->elements[A->num_columns*i + j] -= \
                    base_multiplier*A->elements[A->num_columns*current_row + j];
            }
        }
    }
}

void gauss_eliminate_func(void *args) {
    arg_for_thread_t *thread_data = (arg_for_thread_t *) args;
    int k;
    int dim = thread_data->A->num_columns;
    for (k = 0; k < thread_data->A->num_rows; k++) {
        // Obtain root value of each row before divide stage
        float root_value = thread_data->A->elements[dim*k + k];
        barrier_sync(&root_val_barrier, thread_data->tid, thread_data->num_threads);

        // Divide stage
        divide(thread_data->tid, thread_data->num_threads, k, root_value, thread_data->A);
        barrier_sync(&divide_barrier, thread_data->tid, thread_data->num_threads);

        // Eliminate stage
        eliminate(thread_data->tid, thread_data->num_threads, k, thread_data->A);
        barrier_sync(&eliminate_barrier, thread_data->tid, thread_data->num_threads);
    } 
}

/* FIXME: Write code to perform gaussian elimination using pthreads */
void gauss_eliminate_using_pthreads(Matrix U, unsigned int num_threads)
{
    int i;
    pthread_t *worker = (pthread_t *) malloc (num_threads * sizeof(pthread_t));
    arg_for_thread_t *thread_data;
    for (i = 0; i < num_threads; i++) {
        thread_data = (arg_for_thread_t *) malloc (sizeof(arg_for_thread_t));
        thread_data->tid = i;
        thread_data->num_threads = num_threads;
        thread_data->A = &U;

        if ((pthread_create(&worker[i], NULL, gauss_eliminate_func, (void *)thread_data)) != 0) {
            perror("pthread_create");
            exit(EXIT_FAILURE);
        }
    }
    for (i = 0; i < num_threads; i++) {
        pthread_join(worker[i], NULL);
    }
    return;
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

/* Barrier synchronization implementation */
void barrier_sync(barrier_t *barrier, int tid, int num_threads)
{
    int i;

    sem_wait(&(barrier->counter_sem));
    /* Check if all threads before us, that is num_threads - 1 threads have reached this point. */	  
    if (barrier->counter == (num_threads - 1)) {
        barrier->counter = 0; /* Reset counter value */
        sem_post(&(barrier->counter_sem)); 	 
        /* Signal blocked threads that it is now safe to cross the barrier */
        // printf("Thread number %d is signalling other threads to proceed\n", tid); 
        for (i = 0; i < (num_threads - 1); i++)
            sem_post(&(barrier->barrier_sem));
    } 
    else { /* There are threads behind us */
        barrier->counter++;
        sem_post(&(barrier->counter_sem));
        sem_wait(&(barrier->barrier_sem)); /* Block on the barrier semaphore */
    }

    return;
}