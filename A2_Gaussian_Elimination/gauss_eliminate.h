#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

/* Matrix Structure declaration */
typedef struct {
    unsigned int num_columns;   /* Width of the matrix */ 
    unsigned int num_rows;      /* Height of the matrix */
    float* elements;            /* Pointer to the first element of the matrix */
} Matrix;

typedef struct args_for_div_t
{
    unsigned int tid;
    unsigned int dim;
    unsigned int chunksize;
    unsigned int num_threads;
    unsigned int current_row;
    float root_value;
    float *U;
} args_for_div_t;

typedef struct args_for_eli_t
{
    unsigned int tid;
    unsigned int dim;
    unsigned int current_row;
    unsigned int num_threads;
    float *U;
} args_for_eli_t;

#endif /* _MATRIXMUL_H_ */

