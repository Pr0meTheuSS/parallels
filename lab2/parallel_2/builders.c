// Copyright 2023 Olimpiev Y. Y.
#include <assert.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

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

