// Copyright 2023 Olimpiev Y. Y.
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#include "builders.h"

#include <mpich/mpi.h>
#include <math.h>


void prll_gsl_blas_dgemv(
    CBLAS_TRANSPOSE_t trans, 
    double alpha, 
    gsl_matrix* subA, 
    gsl_vector* x, 
    double beta, 
    gsl_vector* y, 
    gsl_vector* answerPart, 
    int* scounts, 
    int* displs) {

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    gsl_vector* xBlock = gsl_vector_calloc(scounts[rank]);
    assert(xBlock);

    gsl_vector* answerBlock = gsl_vector_calloc(scounts[rank]);
    assert(answerBlock);

    // Send parts of vector x to all threads.
    void* sendbuf = (0 == rank) ? x->data: NULL;
    MPI_Scatterv(sendbuf, scounts, displs, MPI_DOUBLE, xBlock->data, scounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    gsl_vector_set_all(answerPart, 0.0);

    for (int i = 0; i < size; i++) {
        gsl_matrix_view currMatrix = gsl_matrix_submatrix(subA, 0, displs[(i + rank) % size], scounts[i], scounts[i]);

        gsl_blas_dgemv(trans, alpha, &currMatrix.matrix, xBlock, beta, answerBlock);

        gsl_blas_daxpy(1.0, answerBlock, answerPart);

        int left = (rank + size - 1) % size;
        int right = (rank + size + 1) % size;

        MPI_Sendrecv_replace(xBlock->data, scounts[rank], MPI_DOUBLE, left, 1, right, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    void* recv = (0 == rank) ? y->data : NULL;
    MPI_Gatherv(answerPart->data, scounts[rank], MPI_DOUBLE, recv, scounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    gsl_vector_free(xBlock);
    gsl_vector_free(answerBlock);
}

double ConjugateGradientsMethodIteration(
    gsl_matrix* subA, 
    gsl_vector* x, 
    gsl_vector* asnwerPart, 
    gsl_vector* r,
    gsl_vector* z,
    gsl_vector* tmpVec,
    double bNorm,
    int* scounts,
    int* displs) {
    
    double err = 0.0;
    double alpha = 0.0;
    double betta = 0.0;
    double tmp = 0.0;
    
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    prll_gsl_blas_dgemv(CblasNoTrans, 1.0, subA,  z, 0.0, tmpVec, asnwerPart, scounts, displs);
    if (0 == rank) {
        // Calc (r_n, r_n). double tmp <- (r_n, r_n).
        gsl_blas_ddot(r, r, &tmp);
        // Calc (A * z_n, z_n).
        gsl_blas_ddot(tmpVec, z, &alpha);
        // Calc (r_n, r_n) / (A * z_n, z_n).
        alpha = tmp / alpha;
        // Calc x_(n + 1) = x_n + aplha * z_n.
        gsl_blas_daxpy(alpha, z, x);
        // Calc r_(n + 1) = r_n - aplha * (A * z_n).
        gsl_blas_daxpy(-alpha, tmpVec, r);
        // Calc (r_(n + 1), r_(n + 1)).
        gsl_blas_ddot(r, r, &betta);
        // Calc betta_(n + 1) = (r_(n + 1), r_(n + 1)) / (r_n, r_n).
        betta /= tmp;

        // TODO: explain this shit.
        gsl_vector_set_zero(tmpVec);
        gsl_blas_daxpy(betta, z, tmpVec);
        gsl_blas_daxpy(1.0, r, tmpVec);

        gsl_vector_memcpy(z, tmpVec);
        err = gsl_blas_dnrm2(r) / bNorm;
    }

    MPI_Bcast(&err, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return err;
}

gsl_vector* ConjugateGradientsMethod(gsl_matrix* A, gsl_vector* B, gsl_vector* X) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    gsl_vector* tmpVec = NULL;
    gsl_vector* r = NULL;
    gsl_vector* z = NULL;

    int rows = 0;
    int cols = 0;
    int lastChildRows = 0;
    
    double normB = 0.0;
    if (0 == rank) {
        assert(A);
        assert(B);
        assert(X);

        tmpVec = gsl_vector_calloc(B->size);
        assert(tmpVec);

        r = gsl_vector_calloc(B->size);
        assert(r);

        // Calc r = b - Ax. But x = (0), so r = b.
        // TODO: explain this shit.
        gsl_vector_memcpy(r, B);

        z = gsl_vector_calloc(B->size);
        assert(z);

        gsl_vector_memcpy(z, r);
        normB = gsl_blas_dnrm2(B);

        cols = A->size2;
        rows = trunc((double) A->size1 / (double) size);
        lastChildRows = A->size2 - (size - 1) * rows;
    }
    
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&lastChildRows, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Send matrix.
    int* displs = (int*)malloc(size*sizeof(int));
    int* scounts = (int*)malloc(size*sizeof(int));
    assert(displs);
    assert(scounts);

    for (int i = 0; i < size; ++i) {
        scounts[i] = rows * cols;
        if (i == size - 1) {
            scounts[i] = lastChildRows * cols;
        }
        displs[i] = (i == 0) ? 0 : displs[i-1] + scounts[i-1];
    }

    gsl_matrix* subA = gsl_matrix_alloc(scounts[rank] / cols, cols);
    assert(subA);

    gsl_vector* answerPart = gsl_vector_calloc(scounts[rank] / cols);
    assert(answerPart);

    // Send matrix.
    void* dest = (0 == rank) ? A->data : NULL;
    MPI_Scatterv(dest, scounts, displs, MPI_DOUBLE, subA->data, scounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < size; ++i) {
        scounts[i] /= cols;
        displs[i] = (i == 0) ? 0 : displs[i-1] + scounts[i-1];
    }

    double eps = 0.1;
    double err = 0.0;

    do {
        err = ConjugateGradientsMethodIteration(subA, X, answerPart, r, z, tmpVec, normB, scounts, displs);
    } while (eps < err);

    gsl_vector_free(r);
    gsl_vector_free(z);
    gsl_vector_free(tmpVec);

    free(displs);
    free(scounts);
    
    gsl_matrix_free(subA);
    gsl_vector_free(answerPart);

    return X;
}

gsl_vector* CalcGridHeatDistribution(gsl_matrix* gridMatrix) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    gsl_matrix* kernelMatrix = NULL;
    gsl_vector* X = NULL;
    gsl_vector* B = NULL;
    
    if (0 == rank) {
        assert(gridMatrix);
        
        kernelMatrix = BuildKernelMatrix(gridMatrix->size1, gridMatrix->size2);
        assert(kernelMatrix);

        X = BuildAnswerVector(gridMatrix->size1, gridMatrix->size2);
        assert(X);

        B = BuildCoeffsVector(gridMatrix);
        assert(B);
    }

    gsl_vector* ret = ConjugateGradientsMethod(kernelMatrix, B, X);
    gsl_matrix_free(kernelMatrix);
    gsl_vector_free(B);
    return ret;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    #include <unistd.h>
    #include <sys/types.h>
    printf("in thread[%d] pid is(%d)", rank, getpid());
   
    gsl_matrix* gridMatrix = NULL;

    if (0 == rank) {
        size_t colsAmount = 0;
        size_t rowsAmount = 0;

        FILE* in = (argc == 1) ? stdin : fopen(argv[1], "r");
        assert(in);

        if (fscanf(in, "%zu %zu", &rowsAmount, &colsAmount) != 2) {
            perror("Invalid matrix size input.\n");
            return EXIT_FAILURE;
        }
        gridMatrix = ReadGridMatrix(in, rowsAmount, colsAmount);

        if (argc != 1) fclose(in);
    } 

    gsl_vector* result = CalcGridHeatDistribution(gridMatrix);

    if (0 == rank && result) {
        gsl_vector_fprintf(stdout, result, "%4lf ");
        gsl_vector_free(result);
    }
    gsl_matrix_free(gridMatrix);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
