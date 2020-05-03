#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

/* Matrix Structure declaration */
typedef struct {
    unsigned int num_columns;   /* Width of the matrix */ 
    unsigned int num_rows;      /* Height of the matrix */
    float* elements;            /* Pointer to the first element of the matrix */
} Matrix;

/* Structure that defines the barrier */
typedef struct barrier_s {
    sem_t counter_sem;          /* Protects access to the counter */
    sem_t barrier_sem;          /* Signals that barrier is safe to cross */
    int counter;                /* The value itself */
} barrier_t;

typedef struct arg_for_thread_t {
    unsigned int tid;
    unsigned int num_threads;
    Matrix *A;
}arg_for_thread_t;

#endif /* _MATRIXMUL_H_ */

