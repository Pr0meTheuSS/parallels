//Copyright 2023 Olimpiev Y. Y.
#include "builders.h"
#include <assert.h>

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
