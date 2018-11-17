/* @file mmm_mpi.c
 * Authors: Thomas Turner(tdturner@ucdavis.edu)
 *
 */

#include <stdio.h>                      // libc
#include <stdlib.h>                     // libc
#include <string.h>                     // libc
#include <time.h>                       // libc
#include "matrix_checksum.c"
#include <mpi.h>
#include <unistd.h>


#define BILLION                             1000000000
#define TOTAL_CORES                         8
#define MIN(x, y)                           ((y ^ ((x ^ y) & -(x < y))))
static const int MATRIX_SIZE_LOWERBOUND   = 1;
static const int MATRIX_SIZE_UPPERBOUND   = BILLION;
typedef enum{ INIT_STATE, DISTRIBUTE_STATE, IDLE_STATE, COMPUTE_STATE} g_state;

struct start_and_endpoints{
    int sp, ep, sz;
};

typedef struct start_and_endpoints table_lookup_t;

struct matricies{
    double* a, *b, *c;
};

typedef struct matricies matrix_t;

/*
 * Initilize matrix A with Aij = i + j, matrix B with Bij = i + j * 2
 * Input: matrix A, matrix B, N: size of matrix A, B
 * Output: no return, matrix A and B initialized as described
 */
void initialize_matrices(double* matA, double* matB, int N)
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
void print_error_message(const char* error_message)
{
    fprintf(stderr, "%s\n", error_message);
}

/*
 * Parse arguments with following syntax: mmm_omp matrix_size
 * Input: argc: number of arguments, argv: arguments array, N: size of matrix 
 * Output: return 0 if the input follows the syntax, 1 otherwise; N: *OUT* size of matrix
 */
int parse_arguments(int argc, char** argv, int* N)
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

void ikj(matrix_t* m, int start, int end, int N)
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

void mmblock(matrix_t* m, int start, int end, int N){
    int i, ii, j, jj, k, kk;
    int block_size = 16;
    double *matA = m->a;
    double *matB = m->b;
    double *matC = m->c;

    for(i=start; i<end; i += block_size)
        for(j=0; j<N; j += block_size)
            for(k=0; k<N; k += block_size)
                for (ii = i; ii < MIN((i+block_size), end); ii++)
                    for (jj = j; jj < MIN((j+block_size), N); jj++)
                        for (kk = k; kk < MIN((k+block_size), N); kk++)
                            matC[ii*N+jj] += matA[ii*N+kk] * matB[kk*N+jj];
}

int main(int argc, char** argv)
{
    int             ret = 0;
    int             N = 0;
    int             se[3];
    g_state         state = INIT_STATE, next_state;
    struct          timespec start, end;
    double          d_time = 0;
    int             ierr, procid, numprocs, wprocs;
    MPI_Status      status;
    table_lookup_t  *lookup_id=NULL;
    matrix_t        matrix_id;
    MPI_Request     *reqs=NULL, req;
    MPI_Status      *stats=NULL;

    if(MPI_Init(&argc, &argv) != MPI_SUCCESS){
        //error handling
    }
    if(MPI_Comm_rank(MPI_COMM_WORLD, &procid) != MPI_SUCCESS){
        //error handling
    }
    if(MPI_Comm_size(MPI_COMM_WORLD, &numprocs) != MPI_SUCCESS){
        //error handling
    }

    wprocs = numprocs - 1; //number of worker processes

    while(1){

        if(state == INIT_STATE){

            if(procid == 0){
                if (parse_arguments(argc, argv, &N) != 0){
                    ret = 1;
                    goto ERROR;
                }
            }

            if(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD) != MPI_SUCCESS)
                MPI_Abort(MPI_COMM_WORLD, 1);

            if(!(matrix_id.a = malloc(N * N * sizeof(double)))){
                    ret = 1;
                    goto MEMORY_FAIL;
            }

            if(!(matrix_id.b = malloc(N * N * sizeof(double)))){
                ret = 1;
                goto MEMORY_FAIL;
            }

            if(!(matrix_id.c = calloc(N * N,  sizeof(double)))){
                ret = 1;
                goto MEMORY_FAIL;
            }

            if(procid == 0){

                initialize_matrices(matrix_id.a, matrix_id.b, N);
            }

            if(MPI_Bcast(matrix_id.a, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD) != MPI_SUCCESS)
                MPI_Abort(MPI_COMM_WORLD, 1);
            if(MPI_Bcast(matrix_id.b, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD) != MPI_SUCCESS)
                MPI_Abort(MPI_COMM_WORLD, 1);

            if(procid == 0){
                lookup_id = malloc((numprocs)*sizeof(table_lookup_t));
                if(!lookup_id){
                    ret = 1;
                    goto MEMORY_FAIL;
                }
                reqs = malloc(numprocs*sizeof(MPI_Request));
                if(!reqs){
                    ret = 1;
                    goto MEMORY_FAIL;
                }
                stats = malloc(numprocs*sizeof(MPI_Status));
                if(!stats){
                    ret = 1;
                    goto MEMORY_FAIL;
                }
            }


            next_state = DISTRIBUTE_STATE;

        } else if(state == DISTRIBUTE_STATE){
                if(procid == 0){
                    int i;
                    int i_step = N / wprocs;
                    int r = N % wprocs;
                    int prev_e = 0;
                    for (i = 0; i < wprocs; ++i){
                        se[0] = prev_e;
                        se[1] = MIN((prev_e+i_step), N);
                        if(r > 0){
                            r-=1;
                            se[1]+=1;
                        }
                        se[2] = (se[1]-se[0])*N;
                        prev_e = se[1];
                        lookup_id[i+1].sp = se[0];
                        lookup_id[i+1].ep = se[1];
                        lookup_id[i+1].sz = se[2];

                        ierr = MPI_Send(se, 3, MPI_INT, i+1, 0, MPI_COMM_WORLD);
                        if(ierr != MPI_SUCCESS)
                            MPI_Abort(MPI_COMM_WORLD, 1);
                    }

                }
                else{
                    ierr = MPI_Recv(se, 3, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
                    if(ierr != MPI_SUCCESS)
                        MPI_Abort(MPI_COMM_WORLD, 1);

                }
                next_state = IDLE_STATE;

        } else if(state == COMPUTE_STATE){

            if(procid == 0){
                int i, offset;
                int sp, sz;
                clock_gettime(CLOCK_MONOTONIC, &start);
                for(i=1; i<numprocs; ++i){
                    sp = lookup_id[i].sp;
                    sz = lookup_id[i].sz;
                    offset = N * sp;
                    MPI_Irecv(matrix_id.c+offset, sz, MPI_DOUBLE, i, 0,
                              MPI_COMM_WORLD, &reqs[i]);

                }

                MPI_Waitall(wprocs, reqs+1, stats+1);

            }
            else{

                ikj(&matrix_id, se[0], se[1], N);
                ierr = MPI_Isend(matrix_id.c+(se[0]*N), se[2], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &req);
                MPI_Wait( &req, NULL);
            }

            if(procid==0){
                clock_gettime(CLOCK_MONOTONIC, &end);
                d_time = (double) (end.tv_sec - start.tv_sec) + 1.0
                                    * (end.tv_nsec - start.tv_nsec) / BILLION;
                printf("Running time: %f secs\n", d_time);
                // print matrix checksum
                printf("A: %u\n", matrix_checksum(N, matrix_id.a, sizeof(double)));
                printf("B: %u\n", matrix_checksum(N, matrix_id.b, sizeof(double)));
                printf("C: %u\n", matrix_checksum(N, matrix_id.c, sizeof(double)));

            }
            break;

        } else if(state == IDLE_STATE){
            ierr = MPI_Barrier(MPI_COMM_WORLD);
            if(ierr != MPI_SUCCESS)
                MPI_Abort(MPI_COMM_WORLD, 1);
            next_state = COMPUTE_STATE;
        }
        state = next_state;
    }

MEMORY_FAIL:
    if(ret > 0){
        print_error_message("failed to allocate memory!");
    }
ERROR:
    ierr = MPI_Finalize();
    if(ierr != MPI_SUCCESS)
        MPI_Abort(MPI_COMM_WORLD, 1);

    return ret;
}
