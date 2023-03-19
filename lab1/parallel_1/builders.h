// Copyright 2023 Olimpiev Y. Y.
#pragma once
#define DEBUG

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
gsl_matrix* BuildKernelMatrix(size_t rowsAmount, size_t colsAmount);
gsl_vector* BuildAnswerVector(size_t rowsAmount, size_t colsAmount);
gsl_vector* BuildCoeffsVector(gsl_matrix* gridMatrix);
void pretty_gsl_matrix_fprintf(FILE* out, gsl_matrix* matrix, const char* format);
gsl_matrix* ReadGridMatrix(FILE* in, size_t rowsAmount, size_t colsAmount);
