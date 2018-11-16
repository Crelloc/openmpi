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
static const int EXITCODE_FAILURE         = 1;
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

    for(i=start; i<end; ++i){
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
    else{
        wprocs = numprocs -1; //number of worker processes
    }

    while(1){
        if(state == INIT_STATE){
//             printf("state %d, ProcID %d, start=%d, end=%d\n", state,
//                                                         procid, se[0], se[1]);
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
                //printf("state %d, ProcID %d, start=%d, end=%d\n", state,
                //                                        procid, se[0], se[1]);
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
//             printf("state %d, ProcID %d, start=%d, end=%d\n", state,
//                                                        procid, se[0], se[1]);
            if(procid == 0){
                int i, offset;
                int sp, sz;//sp2, ep2, sz2=0;
//                 double tbuf[250000];
                clock_gettime(CLOCK_MONOTONIC, &start);
                for(i=1; i<numprocs; ++i){
                    sp = lookup_id[i].sp;
                    sz = lookup_id[i].sz;
                    offset = N * sp;
//                     printf("frome rank %d: rank %d, start %d, end %d, sz %d, offset %d\n", procid,i ,sp, lookup_id[i].ep, sz, offset);
                    MPI_Irecv(matrix_id.c+offset, sz, MPI_DOUBLE, i, 0,
                              MPI_COMM_WORLD, &reqs[i]);
//                     ierr = MPI_Recv(matrix_id.c+offset, sz, MPI_DOUBLE, i, 0,
//                               MPI_COMM_WORLD, &status);
//                     if(ierr != MPI_SUCCESS){
//                         printf("reveived wrong size\n");
//                         MPI_Abort(MPI_COMM_WORLD, 1);
//                     }
//                     else{
// //                                   for(int i=0; i<N*N; i+=5){
// //                     printf("%f %f %f %f %f\n", matrix_id.c[i],
// //                         matrix_id.c[i+1],matrix_id.c[i+2],matrix_id.c[i+3],matrix_id.c[i+4]
// //                     );
// //                 } putchar('\n');
//                     }
                }

                MPI_Waitall(wprocs, reqs+1, stats+1);

            }
            else{
                //printf("procid %d is computing matrix multiplication\n", procid);
                ikj(&matrix_id, se[0], se[1], N);
//                 printf("from rank %d: rank %d, start %d, end %d, sz %d\n", procid, procid, se[0], se[1],(se[1]-se[0]+1)*N);
                ierr = MPI_Isend(matrix_id.c+(se[0]*N), se[2], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &req);
//                 for(int i=0; i<N*N; i+=5){
//                     printf("%f %f %f %f %f\n", matrix_id.c[i],
//                         matrix_id.c[i+1],matrix_id.c[i+2],matrix_id.c[i+3],matrix_id.c[i+4]
//                     ); 
//                 }
//                 putchar('\n');
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

//                 for(int i=0; i<N*N; i+=5){
//                     printf("%f %f %f %f %f\n", matrix_id.c[i],
//                         matrix_id.c[i+1],matrix_id.c[i+2],matrix_id.c[i+3],matrix_id.c[i+4]
//                     );
//                 }
            }
            break;

        } else if(state == IDLE_STATE){
            //printf("state %d, ProcID %d, start=%d, end=%d\n", state,
            //                                            procid, se[0], se[1]);
            ierr = MPI_Barrier(MPI_COMM_WORLD);
            if(ierr != MPI_SUCCESS)
                MPI_Abort(MPI_COMM_WORLD, 1);
            next_state = COMPUTE_STATE;
        }
        state = next_state;
    }
    //printf("state: break, ProcID %d done\n", procid);

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
