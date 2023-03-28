// Copyright 2023 Olimpiev Y. Y.
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

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

gsl_matrix* ReadGridMatrix(FILE* in, size_t rowsAmount, size_t colsAmount) {
    assert(in);
    
    gsl_matrix* gridMatrix = gsl_matrix_calloc(rowsAmount, colsAmount);
    assert(gridMatrix);

    if (gsl_matrix_fscanf(in, gridMatrix) != 0) {
        perror("Problem with grid matrix reading.\n");
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

double ConjugateGradientsMethodIteration(
    gsl_matrix* A, 
    gsl_vector* x, 
    gsl_vector* r,
    gsl_vector* z,
    gsl_vector* tmpVec,
    double bNorm) {
    double err = 0.0;
    double alpha = 0.0;
    double betta = 0.0;
    double tmp = 0.0;
 
    gsl_blas_dgemv(CblasNoTrans, 1.0, A, z, 0.0, tmpVec);        
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


    double err = 0.0;
    double normB = gsl_blas_dnrm2(B);

    do {
        err = ConjugateGradientsMethodIteration(A, X, r, z, tmpVec, normB);
    } while (eps < err);

    gsl_vector_free(r);
    gsl_vector_free(z);
    gsl_vector_free(tmpVec);
    return X;
}

gsl_matrix* BuildKernelMatrix(size_t rowsAmount, size_t colsAmount) {
    size_t kernelMatrixSize = colsAmount * rowsAmount;
    gsl_matrix* kernelMatrix = gsl_matrix_calloc(kernelMatrixSize, kernelMatrixSize);
    assert(kernelMatrix);

    for (size_t row = 0; row < kernelMatrixSize - colsAmount; row++) {
        // Set three diagonals.
        gsl_matrix_set(kernelMatrix, row, row, -4.0);
        gsl_matrix_set(kernelMatrix, row, row + 1, 1.0);
        gsl_matrix_set(kernelMatrix, row + 1, row, 1.0);

        gsl_matrix_set(kernelMatrix, row + colsAmount, row, 1.0);
        gsl_matrix_set(kernelMatrix, row, row + colsAmount, 1.0);        
    }
    for (size_t row = kernelMatrixSize - colsAmount; row < kernelMatrixSize; row++) {
        // Set three diagonals.
        gsl_matrix_set(kernelMatrix, row, row, -4.0);
        gsl_matrix_set(kernelMatrix, row, row - 1, 1.0);
        gsl_matrix_set(kernelMatrix, row - 1, row, 1.0);

        gsl_matrix_set(kernelMatrix, row - colsAmount, row, 1.0);
        gsl_matrix_set(kernelMatrix, row, row - colsAmount, 1.0);        

    }

    #ifdef DEBUG
        FILE* out = fopen("kernelmatrix.dat", "w");
        assert(out);
        pretty_gsl_matrix_fprintf(out, kernelMatrix, "%lf ");
        fclose(out);
    #endif

    return kernelMatrix;
}

gsl_vector* BuildAnswerVector(size_t rowsAmount, size_t colsAmount) {
    gsl_vector* answerVector = gsl_vector_calloc(rowsAmount * colsAmount);
    assert(answerVector);
    #ifdef DEBUG
        FILE* out = fopen("answervector.dat", "w");
        assert(out);
        gsl_vector_fprintf(out, answerVector, "%lf ");
        fclose(out);
    #endif

    return answerVector;
}


gsl_vector* BuildCoeffsVector(gsl_matrix* gridMatrix) {
    size_t vectorSize = gridMatrix->size1 * gridMatrix->size2;
    gsl_vector* coeffsVector = gsl_vector_calloc(vectorSize);
    assert(coeffsVector);
    
    for (size_t i = 0; i < vectorSize; i++) {
        gsl_vector_set(coeffsVector, i, gsl_matrix_get(gridMatrix, i / gridMatrix->size2, i % gridMatrix->size2));
    }
    
    #ifdef DEBUG
        FILE* out = fopen("coeffsvector.dat", "w");
        assert(out);
        gsl_vector_fprintf(out, coeffsVector, "%lf ");
        fclose(out);
    #endif

    return coeffsVector;
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
