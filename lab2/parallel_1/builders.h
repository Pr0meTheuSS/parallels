// Copyright 2023 Olimpiev Y. Y.
#pragma once
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

gsl_matrix* BuildKernelMatrix(size_t rowsAmount, size_t colsAmount);
gsl_vector* BuildAnswerVector(size_t rowsAmount, size_t colsAmount);
gsl_vector* BuildCoeffsVector(gsl_matrix* gridMatrix);
