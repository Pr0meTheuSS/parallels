/*
 * Created on Sat Apr 14 2023
 *
 * Copyright (c) 2023 Olimpiev Y. Y.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <mpich/mpi.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

void pretty_gsl_matrix_fprintf(FILE *out, gsl_matrix *matrix, const char *format) {
    assert(out);
    assert(matrix);

    for (size_t row = 0; row < matrix->size1; row++) {
        for (size_t col = 0; col < matrix->size2; col++) {
            fprintf(out, format, gsl_matrix_get(matrix, row, col));
        }
        fprintf(out, "\n");
    }
}

gsl_matrix *ReadMatrix(FILE *in) {
    assert(in);
    size_t rowsAmount = 0;
    size_t colsAmount = 0;

    if (fscanf(in, "%zu %zu", &rowsAmount, &colsAmount) != 2) {
        perror("Cannot read width and height of matrix from file.\n");
        return NULL;
    }

    gsl_matrix *gridMatrix = gsl_matrix_calloc(rowsAmount, colsAmount);
    assert(gridMatrix);

    if (gsl_matrix_fscanf(in, gridMatrix) != 0) {
        perror("Error in gsl matrix fscanf.\n");
        return NULL;
    }

    return gridMatrix;
}

void SeparateTasks(int *taskSizes, size_t jobSize, size_t workersAmount) {
    assert(taskSizes);
    size_t taskSize = jobSize / workersAmount;
    size_t unallocJob = jobSize % workersAmount;

    for (size_t i = 0; i < workersAmount; i++) {
        taskSizes[i] = taskSize;
        if (unallocJob) {
            taskSizes[i]++;
            unallocJob--;
        } 
    }
}

int* InitTaskSizes(size_t jobSize, int gridParam) {
    int *taskSizes = (int *)calloc(gridParam, sizeof(taskSizes[0])); if (!taskSizes) {
        perror("Error: calloc failed\n");
        return NULL;
    }
    // Раскидываем строки матрицы.
    SeparateTasks(taskSizes, jobSize, gridParam);
    return taskSizes;
}

int* InitTaskDispls(int* jobSizes, size_t gridParam) { 
    int* displs = (int*) calloc(gridParam, sizeof(displs[0]));
    assert(displs);

    for (size_t i = 0; i < gridParam; i++) {
        displs[i] = (i == 0) ? 0 : jobSizes[i - 1] + displs[i - 1];
    }

    return displs;
}

gsl_matrix* DistributeMatrixesOverTheGrid(gsl_matrix* matrixes[2], MPI_Comm comm, const int gridParam[2]) {
    assert(matrixes);
    assert(matrixes[0]);
    assert(matrixes[1]);

    gsl_matrix_transpose(matrixes[1]);
    MPI_Comm subComm[2];
    gsl_matrix* subMatrix[2] = {NULL, NULL};

    for (size_t matrixIndex = 0; matrixIndex < 2; matrixIndex++) {
        int rank = 0;
        MPI_Comm_rank(comm, &rank);

        int freeCoords[2] = {0, 1};

        // Build sub communicator for rows.
        if (MPI_Cart_sub(comm, freeCoords, &subComm[matrixIndex]) != MPI_SUCCESS) {
            perror("Failed in cart sub\n");
            return NULL;
        }

        int r = 0;
        MPI_Comm_rank(subComm[matrixIndex], &r);

        int* taskSizes = InitTaskSizes(matrixes[matrixIndex]->size1, gridParam[matrixIndex]);
        int* taskDispls = InitTaskDispls(taskSizes, gridParam[matrixIndex]);

        subMatrix[matrixIndex] = gsl_matrix_calloc(taskSizes[r], matrixes[matrixIndex]->size2);
        for (int i = 0; i < gridParam[matrixIndex]; i++) {
            taskSizes[i] *= matrixes[matrixIndex]->size2;
            taskDispls[i] *= matrixes[matrixIndex]->size2;
        }

        MPI_Scatterv(matrixes[matrixIndex]->data + taskDispls[r], taskSizes, taskDispls, MPI_DOUBLE, subMatrix[matrixIndex]->data, taskSizes[r], MPI_DOUBLE, 0, subComm[matrixIndex]);

        freeCoords[0] = 1;
        freeCoords[1] = 0;
    }

    gsl_matrix* transSubB = gsl_matrix_calloc(subMatrix[1]->size2, subMatrix[1]->size1);
    gsl_matrix_transpose_memcpy(transSubB, subMatrix[1]);
 
    gsl_matrix* subResult = gsl_matrix_calloc(subMatrix[0]->size1, transSubB->size2);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, subMatrix[0], transSubB, 0.0, subResult);

    int coords[2];
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    gsl_matrix* result = NULL;


    int size = 0;
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        result = gsl_matrix_calloc(matrixes[0]->size1, matrixes[1]->size2);
        gsl_matrix_view resMatrixView = gsl_matrix_submatrix(result, 
            0, 
            0, 
            subResult->size1,
            subResult->size2);
        gsl_matrix_memcpy(&resMatrixView.matrix, subResult);

        for (int p = 1; p < size; p++) {
            MPI_Cart_coords(comm, p, 2, coords);    

            MPI_Recv(subResult->data, subResult->size1 * subResult->size2, MPI_DOUBLE, p, 1, comm, MPI_STATUS_IGNORE);
            resMatrixView = gsl_matrix_submatrix(result, 
                result->size1 / gridParam[0] * coords[0], 
                result->size2 / gridParam[1] * coords[1], 
                subResult->size1,
                subResult->size2);
            gsl_matrix_memcpy(&resMatrixView.matrix, subResult);
        }
    } else {
        MPI_Send(subResult->data, subResult->size1 * subResult->size2, MPI_DOUBLE, 0, 1, comm);
    }

    return result;
}


gsl_matrix* GridTopologyMatrixMultiplication(gsl_matrix* A, gsl_matrix* B, size_t gridWidth, size_t gridHeight) {
    assert(A);
    assert(B);
    assert(gridWidth > 0);
    assert(gridHeight > 0);

    // Build topolody and communicators.
    MPI_Comm grid_comm;
    int dimSizes[2] = {gridWidth, gridHeight};

    int wrap_around[2] = {0, 0};
    if (MPI_SUCCESS != MPI_Cart_create(MPI_COMM_WORLD, 2, dimSizes, wrap_around, 0, &grid_comm)) {
        perror("Failed in cart create.\n");
        return NULL;
    }

    // matrixes distribution over the grid (A and B).
    gsl_matrix* matrixes[2] = {A, B};
    return DistributeMatrixesOverTheGrid(matrixes, grid_comm, dimSizes);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    gsl_matrix *matrixA = NULL;
    gsl_matrix *matrixB = NULL;

    if (argc < 3) {
        perror("Wrong input. Expected params in command line args (gridWidth gridHeight)");
        return EXIT_FAILURE;
    }

    // params for 2D grid.
    size_t gridWidth = 0; if (sscanf(argv[1], "%zu", &gridWidth) != 1) {
        perror("Wrong input. Cannot read grid width.\n");
        return EXIT_FAILURE;
    }

    size_t gridHeight = 0; if (sscanf(argv[2], "%zu", &gridHeight) != 1) {
        perror("Wrong input. Cannot read grid height.\n");
        return EXIT_FAILURE;
    }

    if (5 == argc) {
        FILE *matrixAFileSrc = fopen(argv[3], "r");
        if (!matrixAFileSrc) {
            perror("Wrong input. Cannot read matrix A from file.\n");
            return EXIT_FAILURE;
        }

        matrixA = ReadMatrix(matrixAFileSrc);
        fclose(matrixAFileSrc);
        if (!matrixA) {
            perror("Wrong matrix A. ReadMatrix return NULL.\n");
            return EXIT_FAILURE;
        }

        FILE *matrixBFileSrc = fopen(argv[4], "r");
        if (!matrixBFileSrc) {
            perror("Wrong input. Cannot read matrix B from file.\n");
            return EXIT_FAILURE;
        }

        matrixB = ReadMatrix(matrixBFileSrc);
        fclose(matrixBFileSrc);
        if (!matrixB) {
            perror("Wrong matrix B. ReadMatrix return NULL.\n");
            return EXIT_FAILURE;
        }
    } else {
        // There is need read matrixes from console, but this feature was not implemented yet.
        perror("Wrong input. Use cli to pass params.\n");
        return EXIT_FAILURE;
    }
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    gsl_matrix *result = GridTopologyMatrixMultiplication(matrixA, matrixB, gridWidth, gridHeight);
    if (result && rank == 0) {
       pretty_gsl_matrix_fprintf(stdout, result, "%lf ");
    }
    gsl_matrix_free(result);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
