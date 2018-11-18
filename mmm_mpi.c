/* @file mmm_mpi.c
 *
 *
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "matrix_checksum.c"
#include <mpi.h>
#include <unistd.h>

#define                                     BILLION 1E9
#define TOTAL_CORES                         8
#define MIN(x, y)                           ((y ^ ((x ^ y) & -(x < y))))
static const int MATRIX_SIZE_LOWERBOUND   = 1;
static const int MATRIX_SIZE_UPPERBOUND   = BILLION;


struct matricies{
    double* a, *b, *c;
};

typedef struct matricies matrix_t;

static void free_matrices(matrix_t *mat){
    free(mat->a);
    free(mat->b);
    free(mat->c);
}

static int alloc_matrices(matrix_t *mat, int N){

    int ret = 0;

    if(!(mat->a = malloc(N * N * sizeof(double)))){
        ret = -1;
        goto MERROR;
    }
    if(!(mat->b = malloc(N * N * sizeof(double)))){
        ret = -1;
        goto MERROR;
    }
    if(!(mat->c = calloc(N * N,  sizeof(double)))){
        ret = -1;
        goto MERROR;
    }

MERROR:

    return ret;
}

//https://www.softwariness.com/articles/monotonic-clocks-windows-and-posix/
static double TimeSpecToSeconds(struct timespec* ts)
{
    return (double)ts->tv_sec + (double)ts->tv_nsec / 1000000000.0;
}
/*
 * Initilize matrix A with Aij = i + j, matrix B with Bij = i + j * 2
 * Input: matrix A, matrix B, N: size of matrix A, B
 * Output: no return, matrix A and B initialized as described
 */
static void initialize_matrices(double* matA, double* matB, int N)
{
    int i = 0;
    int j = 0;
    for (i = 0; i < N; i++)
    {
        j = 0;
        for (j = 0; j < N; j++)
            matA[i*N+j] = i + j; 
    }
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
            matB[i*N+j] = i + j * 2; 
    }
}


/*
 * Print a string to stderr
 * Input: error message
 * Output: no return
 */
static void print_error_message(const char* error_message)
{
    fprintf(stderr, "%s\n", error_message);
}

/*
 * Parse arguments with following syntax: mmm_omp matrix_size
 * Input: argc: number of arguments, argv: arguments array, N: size of matrix 
 * Output: return 0 if the input follows the syntax, 1 otherwise; N: *OUT* size of matrix
 */
static int parse_arguments(int argc, char** argv, int* N)
{
    if (argc != 2)
    {
        print_error_message("Usage: mmm_mpi N");
        return 1;
    }
    *N = atoi(argv[1]);
    if (!((*N) >= MATRIX_SIZE_LOWERBOUND && (*N) <= MATRIX_SIZE_UPPERBOUND))
    {
        print_error_message("Error: wrong matrix order (N > 0)");
        return 1;
    }
    return 0;
}

static void ikj(matrix_t* m, int start, int end, int N)
{
    int i,j,k;
    double r;
    double *matA = m->a;
    double *matB = m->b;
    double *matC = m->c;

    for(i=start; i<end; ++i){
        for(k=0; k<N; ++k){
            r = matA[i*N + k];
            for(j=0; j<N; ++j){
                matC[i*N + j] += (r * matB[k*N + j]);
            }
        }
    }
}

int main(int argc, char** argv)
{
    int             ret = 0, N = 0, se[4]; //se[start, end, size, offset]
    struct          timespec start, end;
    int             procid, numprocs, wprocs;
    MPI_Status      status;
    matrix_t        matrix_id;
    MPI_Request     req[2];

    if(MPI_Init(&argc, &argv) != MPI_SUCCESS){
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if(MPI_Comm_rank(MPI_COMM_WORLD, &procid) != MPI_SUCCESS){
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if(MPI_Comm_size(MPI_COMM_WORLD, &numprocs) != MPI_SUCCESS){
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    wprocs = numprocs - 1; //number of worker processes

/**> initialize section */
    if(procid == 0){
        if (parse_arguments(argc, argv, &N) != 0){
            ret = 1;
            goto ERROR;
        }
    }

    if(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD) != MPI_SUCCESS){
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if(alloc_matrices(&matrix_id, N) < 0){
        print_error_message("failed to allocate memory!");
        ret = 1;
        goto ERROR;
    }

    if(procid == 0){
        initialize_matrices(matrix_id.a, matrix_id.b, N);
    }

    if(MPI_Bcast(matrix_id.a, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD) != MPI_SUCCESS){
        free_matrices(&matrix_id);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if(MPI_Bcast(matrix_id.b, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD) != MPI_SUCCESS){
        free_matrices(&matrix_id);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

/**>distribute section */
    clock_gettime(CLOCK_MONOTONIC, &start);

    if(procid == 0){
        int i;
        int i_step  = N / wprocs;
        int r       = N % wprocs;
        int prev_e  = 0;

        for (i = 0; i < wprocs; ++i){
            se[0] = prev_e;
            se[1] = MIN((prev_e + i_step), N);
            if(r > 0){
                r     -= 1;
                se[1] += 1;
            }
            se[2]  = (se[1] - se[0]) * N;
            se[3]  = se[0] * N;
            prev_e = se[1];

            if(MPI_Send(se, 4, MPI_INT, i+1, 0, MPI_COMM_WORLD) != MPI_SUCCESS){
                free_matrices(&matrix_id);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

    }
    else{
        if(MPI_Recv(se, 4, MPI_INT, 0, 0, MPI_COMM_WORLD, &status) != MPI_SUCCESS){
            free_matrices(&matrix_id);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

/**> compute section */

    if(procid != 0){
        ikj(&matrix_id, se[0], se[1], N);
        MPI_Isend(&se[2], 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &req[0]);
        MPI_Isend(matrix_id.c + se[3], se[2], MPI_DOUBLE, 0, 0,
                         MPI_COMM_WORLD, &req[1]);
        MPI_Waitall(2, req, NULL);
    }
    else{
        int i;
        for(i = 1; i < numprocs; ++i){
            if(MPI_Recv(&se[2], 2, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status) != MPI_SUCCESS){
                free_matrices(&matrix_id);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if(MPI_Recv(matrix_id.c + se[3], se[2], MPI_DOUBLE, i, 0,
                                           MPI_COMM_WORLD, &status) != MPI_SUCCESS){
                    free_matrices(&matrix_id);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    if(procid==0){
#if defined (_TS_)
        printf("%d, %d, %f\n", N, numprocs, TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start) );
#else
        printf("Running time: %f secs\n", TimeSpecToSeconds(&end) - TimeSpecToSeconds(&start));
        printf("A: %u\n", matrix_checksum(N, matrix_id.a, sizeof(double)));
        printf("B: %u\n", matrix_checksum(N, matrix_id.b, sizeof(double)));
        printf("C: %u\n", matrix_checksum(N, matrix_id.c, sizeof(double)));
#endif
    }

    free_matrices(&matrix_id);
ERROR:
    if(MPI_Finalize() != MPI_SUCCESS)
        MPI_Abort(MPI_COMM_WORLD, 1);

    return ret;
}
