/*
 ********************************************************************

 Example 22 (mm_mult_cannon.c)

 Objective           : Matrix Matrix multiplication
 (Using Cartesian Topology, CANNON Algorithm)

 Input               : Read files (mdata1.inp) for first input matrix
 and (mdata2.inp) for second input matrix

 Output              : Result of matrix matrix multiplication on Processor 0.

 Necessary Condition : Number of Processes should be less than
 or equal to 8. Matrices A and B should be
 equally striped. that is Row size and
 Column size should be properly divisible
 by Number of processes used.

 ********************************************************************
 */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <cblas.h>
#include <stdint.h>
#include <mpi.h>
#include <stdlib.h>

#include "util/timer.h"
#include "glog/logging.h"
#include "google/gflags.h"

DEFINE_int32(edge_size, 100, "");

#define NDIMENSIONS 2

typedef struct {
  int Size; /* The number of processors. (Size = q_proc*q_proc)       */
  int p_proc; /* The number of processors in a row (column).  */
  int Row; /* The mesh row this processor occupies.        */
  int Col; /* The mesh column this processor occupies.     */
  int MyRank; /* This processors unique identifier.           */
  MPI_Comm Comm; /* Communicator for all processors in the mesh. */
  MPI_Comm Row_comm; /* All processors in this processors row   .    */
  MPI_Comm Col_comm; /* All processors in this processors column.    */
} MESH_INFO_TYPE;

/* Communication block set up for mesh toplogy */
void SetUp_Mesh(MESH_INFO_TYPE *);

#define MLOG LOG_IF(INFO, grid.MyRank == Root) << "T: " << t.elapsed() << " :: "

int main(int argc, char** argv) {
  int istage, irow, icol, iproc, jproc, index, Proc_Id, Root = 0;
  int A_Bloc_MatrixSize, B_Bloc_MatrixSize;
  int NoofRows_A, NoofCols_A, NoofRows_B, NoofCols_B;
  int NoofRows_BlocA, NoofCols_BlocA, NoofRows_BlocB, NoofCols_BlocB;
  int Local_Index, Global_Row_Index, Global_Col_Index;
  int Matrix_Size[4];
  int source, destination, send_tag, recv_tag;
  float **Matrix_A, **Matrix_B, **Matrix_C;
  float *A_Bloc_Matrix, *B_Bloc_Matrix, *C_Bloc_Matrix;

  float *MatA_array, *MatB_array, *MatC_array;

  MESH_INFO_TYPE grid;
  MPI_Status status;

  MPI_Init(&argc, &argv);

  FLAGS_logtostderr = true;
  FLAGS_log_prefix = false;

  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  Matrix_A = Matrix_B = Matrix_C = NULL;
  MatC_array = NULL;

  NoofCols_A = NoofCols_B = NoofRows_A = NoofRows_B = FLAGS_edge_size;

  Matrix_Size[0] = NoofRows_A;
  Matrix_Size[1] = NoofCols_A;
  Matrix_Size[2] = NoofRows_B;
  Matrix_Size[3] = NoofCols_B;


  /* Set up the MPI_COMM_WORLD and CARTESIAN TOPOLOGY */
  SetUp_Mesh(&grid);

  piccolo::Timer t;

  /* Reading Input */
  if (grid.MyRank == Root) {
    MLOG << "Initializing input.";

    Matrix_A = (float **) malloc(NoofRows_A * sizeof(float *));
    for (irow = 0; irow < NoofRows_A; irow++) {
      Matrix_A[irow] = (float *) malloc(NoofCols_A * sizeof(float));
      for (icol = 0; icol < NoofCols_A; icol++)
        Matrix_A[irow][icol] = 2.0;
    }

    Matrix_B = (float **) malloc(NoofRows_B * sizeof(float *));
    for (irow = 0; irow < NoofRows_B; irow++) {
      Matrix_B[irow] = (float *) malloc(NoofCols_B * sizeof(float *));

      for (icol = 0; icol < NoofCols_B; icol++)
        Matrix_B[irow][icol] = 2.0;
    }

    MatC_array = (float *) malloc(sizeof(float) * NoofRows_A * NoofCols_B);
  }

  MPI_Barrier(grid.Comm);

  if (NoofCols_A != NoofRows_B) {
    if (grid.MyRank == Root) {
      printf("Matrices Dimensions incompatible for Multiplication");
    }
    exit(-1);
  }

  if (NoofRows_A % grid.p_proc != 0 || NoofCols_A % grid.p_proc != 0
      || NoofRows_B % grid.p_proc != 0 || NoofCols_B % grid.p_proc != 0) {

    if (grid.MyRank == Root) {
      printf("Matrices can't be divided among processors equally");
    }
    exit(-1);
  }

  NoofRows_BlocA = NoofRows_A / grid.p_proc;
  NoofCols_BlocA = NoofCols_A / grid.p_proc;

  NoofRows_BlocB = NoofRows_B / grid.p_proc;
  NoofCols_BlocB = NoofCols_B / grid.p_proc;

  A_Bloc_MatrixSize = NoofRows_BlocA * NoofCols_BlocA;
  B_Bloc_MatrixSize = NoofRows_BlocB * NoofCols_BlocB;

  /* Memory allocating for Bloc Matrices */
  A_Bloc_Matrix = (float *) malloc(A_Bloc_MatrixSize * sizeof(float));
  B_Bloc_Matrix = (float *) malloc(B_Bloc_MatrixSize * sizeof(float));

  /* memory for arrangmeent of the data in one dim. arrays before MPI_SCATTER */
  MatA_array = (float *) malloc(sizeof(float) * NoofRows_A * NoofCols_A);
  MatB_array = (float *) malloc(sizeof(float) * NoofRows_B * NoofCols_B);

  /*Rearrange the input matrices in one dim arrays by approriate order*/
  if (grid.MyRank == Root) {

    MLOG << "A: " << NoofRows_A << ", " << NoofCols_A << ", " << A_Bloc_MatrixSize << " :: "
              << "B: " << NoofRows_B << ", " << NoofCols_B << ", " << B_Bloc_MatrixSize;

    MLOG << "Arranging matrices.";

    /* Rearranging Matrix A*/
    for (iproc = 0; iproc < grid.p_proc; iproc++) {
      for (jproc = 0; jproc < grid.p_proc; jproc++) {
        Proc_Id = iproc * grid.p_proc + jproc;
        for (irow = 0; irow < NoofRows_BlocA; irow++) {
          Global_Row_Index = iproc * NoofRows_BlocA + irow;
          for (icol = 0; icol < NoofCols_BlocA; icol++) {
            Local_Index = (Proc_Id * A_Bloc_MatrixSize) + (irow
                * NoofCols_BlocA) + icol;
            Global_Col_Index = jproc * NoofCols_BlocA + icol;
            MatA_array[Local_Index]
                = Matrix_A[Global_Row_Index][Global_Col_Index];
          }
        }
      }
    }

    /* Rearranging Matrix B*/
    for (iproc = 0; iproc < grid.p_proc; iproc++) {
      for (jproc = 0; jproc < grid.p_proc; jproc++) {
        Proc_Id = iproc * grid.p_proc + jproc;
        for (irow = 0; irow < NoofRows_BlocB; irow++) {
          Global_Row_Index = iproc * NoofRows_BlocB + irow;
          for (icol = 0; icol < NoofCols_BlocB; icol++) {
            Local_Index = (Proc_Id * B_Bloc_MatrixSize) + (irow
                * NoofCols_BlocB) + icol;
            Global_Col_Index = jproc * NoofCols_BlocB + icol;
            MatB_array[Local_Index]
                = Matrix_B[Global_Row_Index][Global_Col_Index];
          }
        }
      }
    }

  } /* if loop ends here */
  MPI_Barrier(grid.Comm);

  if (grid.MyRank == 0) {
    MLOG << "Scattering.";
  }

  /* Scatter the Data  to all processes by MPI_SCATTER */
//  MPI_Scatter(MatA_array, A_Bloc_MatrixSize, MPI_FLOAT, A_Bloc_Matrix,
//              A_Bloc_MatrixSize, MPI_FLOAT, 0, grid.Comm);
//
//  MPI_Scatter(MatB_array, B_Bloc_MatrixSize, MPI_FLOAT, B_Bloc_Matrix,
//              B_Bloc_MatrixSize, MPI_FLOAT, 0, grid.Comm);

  for (int i = 0; i < A_Bloc_MatrixSize; ++i) {
    A_Bloc_Matrix[i] = 2.0;
    B_Bloc_Matrix[i] = 2.0;
  }


  MPI_Barrier(grid.Comm);
  /* Do initial arrangement of Matrices */
  if (grid.Row != 0) {
    source = (grid.Col + grid.Row) % grid.p_proc;
    destination = (grid.Col + grid.p_proc - grid.Row) % grid.p_proc;
    recv_tag = 0;
    send_tag = 0;
    MPI_Sendrecv_replace(A_Bloc_Matrix, A_Bloc_MatrixSize, MPI_FLOAT,
                         destination, send_tag, source, recv_tag,
                         grid.Row_comm, &status);
  }
  if (grid.Col != 0) {
    source = (grid.Row + grid.Col) % grid.p_proc;
    destination = (grid.Row + grid.p_proc - grid.Col) % grid.p_proc;
    recv_tag = 0;
    send_tag = 0;
    MPI_Sendrecv_replace(B_Bloc_Matrix, B_Bloc_MatrixSize, MPI_FLOAT,
                         destination, send_tag, source, recv_tag,
                         grid.Col_comm, &status);
  }

  /* Allocate Memory for Bloc C Array */
  C_Bloc_Matrix = (float *) malloc(NoofRows_BlocA * NoofCols_BlocB
      * sizeof(float));
  for (index = 0; index < NoofRows_BlocA * NoofCols_BlocB; index++)
    C_Bloc_Matrix[index] = 0;

  t.Reset();
  MPI_Barrier(grid.Comm);
  /* The main loop */

  send_tag = 0;
  recv_tag = 0;
  for (istage = 0; istage < grid.p_proc; istage++) {
    if (grid.MyRank == 0) {
      MLOG << "Iterating... " << istage << " of " << grid.p_proc;
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                NoofRows_BlocA, NoofCols_BlocB, NoofCols_BlocA,
                1,
                A_Bloc_Matrix, NoofRows_BlocA,
                B_Bloc_Matrix, NoofRows_BlocB,
                1,
                C_Bloc_Matrix, NoofCols_BlocA);

    /* Move Bloc of Matrix A by one position left with wraparound */
    source = (grid.Col + 1) % grid.p_proc;
    destination = (grid.Col + grid.p_proc - 1) % grid.p_proc;
    MPI_Sendrecv_replace(A_Bloc_Matrix, A_Bloc_MatrixSize, MPI_FLOAT,
                         destination, send_tag, source, recv_tag,
                         grid.Row_comm, &status);

    /* Move Bloc of Matrix B by one position upwards with wraparound */
    source = (grid.Row + 1) % grid.p_proc;
    destination = (grid.Row + grid.p_proc - 1) % grid.p_proc;
    MPI_Sendrecv_replace(B_Bloc_Matrix, B_Bloc_MatrixSize, MPI_FLOAT,
                         destination, send_tag, source, recv_tag,
                         grid.Col_comm, &status);
  }

  MPI_Barrier(grid.Comm);

  if (grid.MyRank == Root) {
    MLOG << "Gather.";
  }

  /* Gather output block matrices at processor 0 */
  MPI_Gather(C_Bloc_Matrix, NoofRows_BlocA * NoofCols_BlocB, MPI_FLOAT,
             MatC_array, NoofRows_BlocA * NoofCols_BlocB, MPI_FLOAT, Root,
             grid.Comm);

  /* Memory for output global array for OutputMatrix_C after rearrangement */
  if (grid.MyRank == Root) {
    Matrix_C = (float **) malloc(NoofRows_A * sizeof(float *));
    for (irow = 0; irow < NoofRows_A; irow++)
      Matrix_C[irow] = (float *) malloc(NoofCols_B * sizeof(float));
  }

  /* Rearranging the output matrix in a array by approriate order  */
  if (grid.MyRank == Root) {
    for (iproc = 0; iproc < grid.p_proc; iproc++) {
      for (jproc = 0; jproc < grid.p_proc; jproc++) {
        Proc_Id = iproc * grid.p_proc + jproc;
        for (irow = 0; irow < NoofRows_BlocA; irow++) {
          Global_Row_Index = iproc * NoofRows_BlocA + irow;
          for (icol = 0; icol < NoofCols_BlocB; icol++) {
            Local_Index = (Proc_Id * NoofRows_BlocA * NoofCols_BlocB) + (irow
                * NoofCols_BlocB) + icol;
            Global_Col_Index = jproc * NoofCols_BlocB + icol;
            Matrix_C[Global_Row_Index][Global_Col_Index]
                = MatC_array[Local_Index];
          }
        }
      }
    }
    printf("-----------MATRIX MULTIPLICATION RESULTS --------------\n");
    printf("Processor %d, Matrix C : Dimension %d * %d : \n", grid.MyRank,
           NoofRows_A, NoofCols_B);
    for (irow = 0; irow < 4; irow++) {
      for (icol = 0; icol < 4; icol++)
        printf("%.3f ", Matrix_C[irow][icol]);
      printf("\n");
    }
  }

  MPI_Finalize();
  return 0;
}

/* Function : Finds communication information suitable to mesh topology  */
/*            Create Cartesian topology in two dimnesions                */
void SetUp_Mesh(MESH_INFO_TYPE *grid) {

  int Periods[2]; /* For Wraparound in each dimension.*/
  int Dimensions[2]; /* Number of processors in each dimension.*/
  int Coordinates[2]; /* processor Row and Column identification */
  int Remain_dims[2]; /* For row and column communicators */

  /* MPI rank and MPI size */
  MPI_Comm_size(MPI_COMM_WORLD, &(grid->Size));
  MPI_Comm_rank(MPI_COMM_WORLD, &(grid->MyRank));

  /* For square mesh */
  grid->p_proc = (int) sqrt((float) grid->Size);
  if (grid->p_proc * grid->p_proc != grid->Size) {
    if (grid->MyRank == 0) {
      printf("Number of Processors should be perfect square\n");
    }
    exit(-1);
  }

  Dimensions[0] = Dimensions[1] = grid->p_proc;

  /* Wraparound mesh in both dimensions. */
  Periods[0] = Periods[1] = 1;

  /*  Create Cartesian topology  in two dimnesions and  Cartesian
   decomposition of the processes   */
  MPI_Cart_create(MPI_COMM_WORLD, NDIMENSIONS, Dimensions, Periods, 1, &(grid->Comm));
  MPI_Cart_coords(grid->Comm, grid->MyRank, NDIMENSIONS, Coordinates);

  grid->Row = Coordinates[0];
  grid->Col = Coordinates[1];

  /*Construction of row communicator and column communicators
   (use cartesian row and columne machanism to get Row/Col Communicators)  */

  Remain_dims[0] = 0;
  Remain_dims[1] = 1;

  /*The output communicator represents the column containing the process */
  MPI_Cart_sub(grid->Comm, Remain_dims, &(grid->Row_comm));

  Remain_dims[0] = 1;
  Remain_dims[1] = 0;

  /*The output communicator represents the row containing the process */
  MPI_Cart_sub(grid->Comm, Remain_dims, &(grid->Col_comm));
}
