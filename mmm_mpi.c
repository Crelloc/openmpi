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

#define BILLION                             1000000000
#define TOTAL_CORES                         8
static const int EXITCODE_FAILURE         = 1;
static const int MATRIX_SIZE_LOWERBOUND   = 1;
static const int MATRIX_SIZE_UPPERBOUND   = BILLION;
typedef enum{ INIT_STATE, DISTRIBUTE_STATE, IDLE_STATE, COMPUTE_STATE} g_state;

struct start_and_endpoints{
    int sp, ep;
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
        print_error_message("Usage: ./mmm_mpi N");
        return 1;
    }
    *N = atoi(argv[1]);
    if (!((*N) >= MATRIX_SIZE_LOWERBOUND && (*N) <= MATRIX_SIZE_UPPERBOUND))
    {
        print_error_message("Error: wrong matrix order (0 < N <= BILLION)");
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

    for(i=start; i<end+1; ++i){
        for(k=0; k<N; ++k){
            r = matA[i*N + k];
            for(j=0; j<N; ++j){
                matC[i*N + j] += (r * matB[k*N + j]);
            }
        }
    }
}

// static void build_mpi_type(MPI_Datatype *mpi_matrix_t){
// 
//     MPI_Datatype type[3] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
//     int blocklen[1] = {3};
// 
//     MPI_Aint offsets[3];
//     offsets[0] = offsetof(matrix_t, a);
//     offsets[1] = offsetof(matrix_t, b);
//     offsets[2] = offsetof(matrix_t, c);
//     MPI_Type_create_struct(3, blocklen, offsets, type, mpi_matrix_t);
//     MPI_Type_commit(mpi_matrix_t);
// }

int main(int argc, char** argv)
{
    int             ret = 0;
    int             N = 0;
    int             se[2];
    g_state         state = INIT_STATE, next_state;
    struct          timespec start, end;
    double          d_time = 0;
    double          *buf;
    int             ierr, procid, numprocs;
    MPI_Status      status;
    table_lookup_t  *lookup_id=NULL;
    matrix_t        matrix_id;

    ierr = MPI_Init(&argc, &argv);
    if(ierr != MPI_SUCCESS){
        //error handling
    }
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &procid);
    if(ierr != MPI_SUCCESS){
        //error handling
    }
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    if(ierr != MPI_SUCCESS){
        //error handling
    }
    if(numprocs < 2){
        printf("ERROR: Number of processes is less than 2!\n");
        return MPI_Abort(MPI_COMM_WORLD, 1);
    }

    printf("Hello world! I'm process %i out of %i processes\n", 
         procid, numprocs);

    while(1){
        if(state == INIT_STATE){
            printf("state %d, ProcID %d, start=%d, end=%d\n", state,
                                                        procid, se[0], se[1]);
            if(procid == 0){
                if (parse_arguments(argc, argv, &N) != 0){
                    ret = EXITCODE_FAILURE;
                    goto ERROR;
                }
            }

            ierr = MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if(ierr != MPI_SUCCESS)
                MPI_Abort(MPI_COMM_WORLD, 1);

            if(procid == 0){
                lookup_id = malloc((numprocs)*sizeof(table_lookup_t));
                if(!lookup_id){
                    ret = 1;
                    goto MEMORY_FAIL;
                }
                buf = malloc(N * N * sizeof(double));
                if(!buf){
                    ret = 1;
                    goto MEMORY_FAIL;
                }
            }

            matrix_id.a = malloc(N * N * sizeof(double));
            if(!matrix_id.a){
                ret = 1;
                goto MEMORY_FAIL;
            }
            matrix_id.b = malloc(N * N * sizeof(double));
            if(!matrix_id.b){
                ret = 1;
                goto MEMORY_FAIL;
            }
            matrix_id.c = calloc(N * N,  sizeof(double));
            if(!matrix_id.c){
                ret = 1;
                goto MEMORY_FAIL;
            }

            initialize_matrices(matrix_id.a, matrix_id.b, N);
            next_state = DISTRIBUTE_STATE;

        } else if(state == DISTRIBUTE_STATE){
                printf("state %d, ProcID %d, start=%d, end=%d\n", state,
                                                        procid, se[0], se[1]);
                if(procid == 0){
                    int i;
                    int i_step = N / (numprocs - 1);
                    int r = N % (numprocs - 1);
                    for (i = 0; i < (numprocs - 1); ++i){
                        se[0] = i_step*i;
                        se[1] = i_step*(i+1) - 1;
                        lookup_id[i+1].sp = se[0];
                        lookup_id[i+1].ep = se[1];
                        if((i + 1) != (numprocs - 1)){
                            ierr =MPI_Send(se, 2, MPI_INT, i+1, 0, MPI_COMM_WORLD);
                            if(ierr != MPI_SUCCESS)
                                MPI_Abort(MPI_COMM_WORLD, 1);
                        }
                        else{
                            se[1] += r;
                            ierr = MPI_Send(se, 2, MPI_INT, i+1, 0, MPI_COMM_WORLD);
                            if(ierr != MPI_SUCCESS)
                                MPI_Abort(MPI_COMM_WORLD, 1);
                        }
                    }

                }
                else{
                    ierr = MPI_Recv(se, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
                    if(ierr != MPI_SUCCESS)
                        MPI_Abort(MPI_COMM_WORLD, 1);
                }
                next_state = IDLE_STATE;

        } else if(state == COMPUTE_STATE){
            printf("state %d, ProcID %d, start=%d, end=%d\n", state,
                                                        procid, se[0], se[1]);
            if(procid == 0){
                int c = 1;
                int sp, ep, id;

                clock_gettime(CLOCK_MONOTONIC, &start);
                while(c < numprocs){
                    printf("root waiting for %d processes\n", c);
                    ierr = MPI_Recv(buf, N*N, MPI_DOUBLE,
                                    MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
                    if(ierr == MPI_SUCCESS){

                        for(id=0; id<(numprocs-1); ++id){
                            sp = lookup_id[id+1].sp;
                            ep = lookup_id[id+1].ep;
                            for(c=sp; c<=ep; c++){
                                memcpy(matrix_id.c+(c*N), buf+(c*N),
                                                                sizeof(double)*(N));
                            }
                        }
                        ++c;
                    }
                    else MPI_Abort(MPI_COMM_WORLD, 1);
                }


                clock_gettime(CLOCK_MONOTONIC, &end);
                d_time = (double) (end.tv_sec - start.tv_sec) + 1.0
                                    * (end.tv_nsec - start.tv_nsec) / BILLION;
                printf("Running time: %f secs\n", d_time);
                // print matrix checksum
                printf("A: %u\n", matrix_checksum(N, matrix_id.a, sizeof(double)));
                printf("B: %u\n", matrix_checksum(N, matrix_id.b, sizeof(double)));
                printf("C: %u\n", matrix_checksum(N, matrix_id.c, sizeof(double)));

            }
            else{
                printf("procid %d is computing matrix multiplication\n", procid);
                ikj(&matrix_id, se[0], se[1], N);
                ierr = MPI_Send(matrix_id.c, N*N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
                if(ierr != MPI_SUCCESS)
                    MPI_Abort(MPI_COMM_WORLD, 1);
                printf("procid %d done computing matrix multiplication\n", procid);

            }

            break;

        } else if(state == IDLE_STATE){
            printf("state %d, ProcID %d, start=%d, end=%d\n", state,
                                                        procid, se[0], se[1]);
            ierr = MPI_Barrier(MPI_COMM_WORLD);
            if(ierr != MPI_SUCCESS)
                MPI_Abort(MPI_COMM_WORLD, 1);
            next_state = COMPUTE_STATE;
        }
        state = next_state;
    }
    printf("state: break, ProcID %d, start=%d, end=%d\n", procid, se[0], se[1]);

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
