// Copyright 2023 Olimpiev Y. Y.
// TODO: работает правильно на одном потоке, надо посмотреть что происходит при большем числе потоков.
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#include "builders.h"

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

gsl_matrix* ReadGridMatrix(FILE* in, size_t rowsAmount, size_t colsAmount) {
    assert(in);
    
    gsl_matrix* gridMatrix = gsl_matrix_calloc(rowsAmount, colsAmount);
    assert(gridMatrix);

    if (gsl_matrix_fscanf(in, gridMatrix) != 0) {
        perror("Problem with grid matrix reading.\n");
        return NULL;
    }

    return gridMatrix;
}

void openmp_gsl_blas_dgemv(
    CBLAS_TRANSPOSE_t trans, 
    double alpha, 
    gsl_matrix* A, 
    gsl_vector* x, 
    double beta, 
    gsl_vector* res,
    size_t* jobSizes,
    size_t* displs) {
        assert(A);
        assert(x);
        assert(res);
        assert(jobSizes);
        assert(displs);


        {
            int rank = omp_get_thread_num();
            gsl_vector* resPart = gsl_vector_calloc(jobSizes[rank]);
            assert(resPart);

            gsl_matrix_view subMatrixA = gsl_matrix_submatrix(A, displs[rank], 0, jobSizes[rank], A->size2);
            gsl_blas_dgemv(trans, alpha, &subMatrixA.matrix, x, beta, resPart);

            memcpy(res->data + displs[rank], resPart->data, sizeof(double) * jobSizes[rank]);
            gsl_vector_free(resPart);
        }
}

// void openmp_gsl_blas_ddot(gsl_vector* x, gsl_vector* y, double* res, size_t* jobSizes, size_t* displs) {
//     assert(x);
//     assert(y);
//     assert(res);
//     double resPart = 0.0;

//     {
//         gsl_vector_view subX = gsl_vector_subvector(x, displs[rank], jobSizes[rank]);
//         gsl_vector_view subY = gsl_vector_subvector(y, displs[rank], jobSizes[rank]);

//         gsl_blas_ddot(&subX.vector, &subY.vector, &resPart);

//         #pragma omp atomic
//         *res += resPart;
//     }
// }

// void openmp_gsl_blas_daxpy(double alpha, gsl_vector* x, gsl_vector* y, size_t* jobSizes, size_t* displs) {
//     assert(x);
//     assert(y);
//     assert(jobSizes);
//     assert(displs);

//     {
//         gsl_vector_view subX = gsl_vector_subvector(x, displs[rank], jobSizes[rank]);
//         gsl_vector_view subY = gsl_vector_subvector(y, displs[rank], jobSizes[rank]);

//         gsl_blas_daxpy(alpha, &subX.vector, &subY.vector);
//         memcpy(y->data + displs[rank], subY.vector.data, sizeof(double) * jobSizes[rank]);
//     }
// }

double ConjugateGradientsMethodIteration(
    gsl_matrix* A, 
    gsl_vector* x, 
    gsl_vector* r,
    gsl_vector* z,
    gsl_vector* tmpVec,
    double bNorm,
    size_t* jobSizes,
    size_t* displs
    ) {
    double err = 0.0;
    double alpha = 0.0;
    double betta = 0.0;
    double tmp = 0.0;
    
    int rank = omp_get_thread_num();
    gsl_vector* resPart = gsl_vector_calloc(jobSizes[rank]);
    assert(resPart);

    gsl_matrix_view subMatrixA = gsl_matrix_submatrix(A, displs[rank], 0, jobSizes[rank], A->size2);
    gsl_blas_dgemv(CblasNoTrans, 1.0, &subMatrixA.matrix, z, 0.0, resPart);

    memcpy(tmpVec->data + displs[rank], resPart->data, sizeof(double) * jobSizes[rank]);
    gsl_vector_free(resPart);

    #pragma omp single 
    {
    // Calc (r_n, r_n). double tmp <- (r_n, r_n)
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

    gsl_vector_set_zero(tmpVec);
    gsl_blas_daxpy(betta, z, tmpVec);
    gsl_blas_daxpy(1.0, r, tmpVec);

    gsl_vector_memcpy(z, tmpVec);
    err = gsl_blas_dnrm2(r) / bNorm;
    }
 
    return err;
}

gsl_vector* ConjugateGradientsMethod(gsl_matrix* A, gsl_vector* B, gsl_vector* X) {
    assert(A);
    assert(B);
    assert(X);

    double eps = 0.00001;

    gsl_vector* tmpVec = gsl_vector_calloc(B->size);
    assert(tmpVec);

    gsl_vector* r = gsl_vector_calloc(B->size);
    assert(r);

    // Calc r = b - Ax. But x = (0), so r = b.
    gsl_vector_memcpy(r, B);
    gsl_vector* z = gsl_vector_calloc(B->size);
    assert(z);

    gsl_vector_memcpy(z, r);

    // Произвольное относительно большое для значения ошибки число.
    double err = 100.0;
    double normB = gsl_blas_dnrm2(B);

    size_t numThreads = omp_get_max_threads();
    printf("max threads is:%lu\n", numThreads);

    size_t* jobSizes = (size_t*)calloc(numThreads, sizeof(size_t));
    assert(jobSizes);

    size_t* displs = (size_t*)calloc(numThreads, sizeof(size_t));
    assert(displs);

    size_t rowsPerThread = A->size1 / numThreads;

    for (size_t i = 0; i < numThreads; i++) {
        jobSizes[i] = rowsPerThread;
    }

    size_t unallocatedJobSize = A->size1 % numThreads;

    // Распределяем неподеленную работу между исполнителями.
    for (size_t i = 0; unallocatedJobSize; i++, unallocatedJobSize--) {
        jobSizes[i]++;
    }

    displs[0] = 0;
    for (size_t i = 1; i < numThreads; i++) {
        displs[i] = displs[i - 1] + jobSizes[i - 1];
    }
    time_t start = time(0);

    # pragma omp parallel
    {
        printf("Threads amount: %d\n", omp_get_thread_num());

        while(err > eps) {
            err = ConjugateGradientsMethodIteration(A, X, r, z, tmpVec, normB, jobSizes, displs);
        }
    }

    time_t finish = time(0);

    printf("%ld\n", finish - start);

    gsl_vector_free(r);
    gsl_vector_free(z);
    gsl_vector_free(tmpVec);
    return X;
}

gsl_vector* CalcGridHeatDistribution(gsl_matrix* gridMatrix) {
    assert(gridMatrix);
    
    gsl_matrix* kernelMatrix = BuildKernelMatrix(gridMatrix->size1, gridMatrix->size2);
    assert(kernelMatrix);

    gsl_vector* X = BuildAnswerVector(gridMatrix->size1, gridMatrix->size2);
    assert(X);

    gsl_vector* B = BuildCoeffsVector(gridMatrix);
    assert(B);

    gsl_vector* ret = ConjugateGradientsMethod(kernelMatrix, B, X);

    gsl_matrix_free(kernelMatrix);
    gsl_vector_free(B);

    return ret;
}

int main(int argc, char* argv[]) {
    size_t colsAmount = 0;
    size_t rowsAmount = 0;

    gsl_matrix* gridMatrix = NULL;
    FILE* in = (argc == 1) ? stdin : fopen(argv[1], "r");
    assert(in);

    if (fscanf(in, "%zu %zu", &rowsAmount, &colsAmount) != 2) {
        perror("Invalid matrix size input.\n");
        return EXIT_FAILURE;
    }
    gridMatrix = ReadGridMatrix(in, rowsAmount, colsAmount);

    if (argc != 1) fclose(in);

    gsl_vector* result = CalcGridHeatDistribution(gridMatrix);
    if (result) {
        gsl_vector_fprintf(stdout, result, "%4lf ");
        gsl_vector_free(result);
    }
    gsl_matrix_free(gridMatrix);

    return EXIT_SUCCESS;
}
