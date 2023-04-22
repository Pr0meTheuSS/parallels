// Copyright 2023 Olimpiev Y. Y.
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <mpich/mpi.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

void pretty_gsl_matrix_fprintf(FILE* out, gsl_matrix* matrix, const char* format) {
    assert(out);
    assert(matrix);

    for (size_t row = 0; row < matrix->size1; row++) {
        for (size_t col = 0; col < matrix->size2; col++) {
            fprintf(out, format, gsl_matrix_get(matrix, row, col));
        }
        fprintf(out, "\n");
    }
}

gsl_matrix* ReadMatrix(FILE* in) {
    size_t rowsAmount = 0;
    size_t colsAmount = 0;
    assert(in);

    if (fscanf(in, "%zu %zu", &rowsAmount, &colsAmount) != 2) {
        perror("Cannot read width and height of matrix from file.\n");
        return NULL;
    }    

    gsl_matrix* gridMatrix = gsl_matrix_calloc(rowsAmount, colsAmount);
    assert(gridMatrix);

    if (gsl_matrix_fscanf(in, gridMatrix) != 0) {
        perror("Error in gsl matrix fscanf.\n");
        return NULL;
    }

    #ifdef DEBUG
        FILE* out = fopen("gridmatrix.dat", "w");
        assert(out);
        pretty_gsl_matrix_fprintf(out, gridMatrix, "%lf ");
        fclose(out);
    #endif

    return gridMatrix;
}

gsl_matrix* GridTopologyMatrixMultiplication(const gsl_matrix* A, const gsl_matrix* B, size_t gridWidth, size_t gridHeight) {
    assert(A);
    assert(B);
    assert(gridWidth > 0);
    assert(gridHeight > 0);
    
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // TODO : 
    // Build tolology and communicators.
    MPI_Comm grid_comm, row_comm, col_comm ;
    int free_coords[2];
    int dim_sizes[2] = {gridWidth, gridHeight};
    
    printf("%d %d\n", dim_sizes[0], dim_sizes[1]);
    
    int wrap_around[2] = {1, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim_sizes, wrap_around, 0, &grid_comm);

    free_coords[0] = 0;
    free_coords[1] = 1;
    MPI_Cart_sub(grid_comm, free_coords, &row_comm);
    free_coords[0] = 1;
    free_coords[1] = 0;
    MPI_Cart_sub(grid_comm, free_coords, &col_comm);

    // Send A by (x, o).
    int r = 0;
    MPI_Comm_rank(row_comm, &r);
    if (r == 0) {
        MPI_Bcast(A->data, A->size1 * A->size2, MPI_DOUBLE, 0, row_comm);
    }

    // Send B by (0, y).
    MPI_Comm_rank(col_comm, &r);
    if (r == 0) {
        MPI_Bcast(B->data, B->size1 * B->size2, MPI_DOUBLE, 0, col_comm);
    }

    // Bcast submatrixes of A in dim x.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dest_rank = 0;
    int src_rank = 0;        
    MPI_Cart_shift(grid_comm, 0, 1, &rank, );
    // Bcast submatrixes of B.

    // Calc parts of result A * B.

    // Gather parts of result into root executor and return it.
    // TODO: remove stab.
    return NULL;
}

int main(int argc, char* argv[]) {
    // params for 2D grid.
    size_t gridHeight = 0;
    size_t gridWidth = 0;
    gsl_matrix* matrixA = NULL;
    gsl_matrix* matrixB = NULL;
    
    if (argc < 3) {
        perror("Wrong input. Expected params in command line args (gridWidth gridHeight)");
        exit(0);
    }

    if (sscanf(argv[1], "%zu", &gridWidth) != 1) {
        perror("Wrong input. Cannot read grid width.\n");
        exit(1);
    }

    if (sscanf(argv[2], "%zu", &gridHeight) != 1) {
        perror("Wrong input. Cannot read grid height.\n");
        exit(1);
    }

    // If matrixes files gave from command line args:
    if (5 == argc) {
        FILE* matrixAFileSrc = fopen(argv[3], "r");
        if (!matrixAFileSrc) {
            perror("Wrong input. Cannot read matrix A from file.\n");
            exit(1);
        }

        matrixA = ReadMatrix(matrixAFileSrc);
        fclose(matrixAFileSrc);
        if (!matrixA) {
            perror("Wrong matrix A. ReadMatrix return NULL.\n");
            exit(1);
        } 

        FILE* matrixBFileSrc = fopen(argv[4], "r");
        if (!matrixBFileSrc) {
            perror("Wrong input. Cannot read matrix B from file.\n");
            exit(1);
        }

        matrixB = ReadMatrix(matrixBFileSrc);
        fclose(matrixBFileSrc);
        if (!matrixB) {
            perror("Wrong matrix B. ReadMatrix return NULL.\n");
            exit(1);
        } 
    } else {
        // There is need read matrixes from console, but this feature was not implemented yet.
        perror("Wrong input. Use command line args to pass params.\n");
        exit(0);
    }

    MPI_Init(&argc, &argv);

    gsl_matrix* result = GridTopologyMatrixMultiplication(matrixA, matrixB, gridWidth, gridHeight);
    if (result) {
        pretty_gsl_matrix_fprintf(stdout, result, "%lf ");
    }
    gsl_matrix_free(result);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

