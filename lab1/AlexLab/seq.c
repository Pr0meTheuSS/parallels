#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#define CALCULATIN_GERROR 0.00001

#define N_x 128
#define N_y 128

double* getVector(size_t sizeVector) {
    double* vector = calloc(sizeVector, sizeof(double));
    if (!vector) return NULL;
    for (size_t i = 0; i < sizeVector; ++i) {
        vector[i] = 35 * (i % 2);
    }
    return vector;
}

bool getInitData(size_t Nx, size_t Ny, size_t* size, double** A, double** b) {
    size_t rowsA = Ny * Nx;
    size_t columnsA = Ny * Nx;
    (*A) = calloc(rowsA * columnsA, sizeof(double));
    double* a = *A;
    for (size_t i = 0; i < rowsA; ++i) {
        a[i * columnsA + i] = -4;
        if (i%Nx) a[i * columnsA + (i - 1)] = 1;
        if ((i + 1) % Nx) a[i * columnsA + (i + 1)] = 1;
        if (i < rowsA - Nx) a[i * columnsA + (Nx + i)] = 1;
        if (i > Nx - 1) a[i * columnsA + (i - Nx)] = 1;
    }

    *b = getVector(rowsA);
    if (!(*b)) {
        free(*A);
        return false;
    }
    (*size) = Nx * Ny;
    return true;
}

void printArray(double* array, size_t rows, size_t columns) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < columns; ++j) {
            printf("%lf ", array[i * columns + j]);
        }
        printf("\n");
    }
}

double* getX0(size_t N) {
    double* result = calloc(N, sizeof(double));
    if (!result) return NULL;
    return result;
}

double* multMatrix(double* A, size_t rowsA, size_t columnsA, double* B, size_t rowsB, size_t columnsB) {

    size_t rowsRes = rowsA;
    size_t columnsRes = columnsB;
    if (columnsA != rowsB) return NULL;
    double* result = calloc(rowsRes * columnsRes, sizeof(double));
    if (!result) return NULL;

    for (size_t i = 0; i < rowsRes; ++i) {
        double* res = result + i * columnsRes;
        for (size_t k = 0; k < columnsA; ++k) {
            const double* b = B + k * columnsB;
            double a = A[i * columnsA + k];
            for (size_t j = 0; j < columnsRes; ++j) {
                res[j] += a * b[j];
            }
        }
    }
    return result;
}

double* multByValue(double value, double* A, size_t rowsA, size_t columnsA) {
    double* result = calloc(rowsA * columnsA, sizeof(double));
    if (!result) return NULL;
    for (size_t i = 0; i < rowsA; ++i) {
        for (size_t j = 0; j < columnsA; ++j) {
            result[i * columnsA + j] = value * A[i * columnsA + j];
        }
    }
    return result;
}

double* sumMatrix(double* A, size_t rowsA, size_t columnsA, double* B, size_t rowsB, size_t columnsB) {
    if (rowsA != rowsB || columnsA != columnsB) return NULL;
    double* result = calloc(rowsA * columnsA, sizeof(double));
    if (!result) return NULL;
    for (size_t i = 0; i < rowsA; ++i) {
        for (size_t j = 0; j < columnsA; ++j) {
            result[i * columnsA + j] = A[i * columnsA + j] + B[i * columnsB + j];
        }
    }
    return result;
}

double* subMatrix(double* A, size_t rowsA, size_t columnsA, double* B, size_t rowsB, size_t columnsB) {
    if (rowsA != rowsB || columnsA != columnsB) return NULL;
    double* result = calloc(rowsA * columnsA, sizeof(double));
    if (!result) return NULL;
    for (size_t i = 0; i < rowsA; ++i) {
        for (size_t j = 0; j < columnsA; ++j) {
            result[i * columnsA + j] = A[i * columnsA + j] - B[i * columnsB + j];
        }
    }
    return result;
}

double nonScalarProduct(double* v1, size_t sizeV1, double* v2, size_t sizeV2) {
    if (sizeV1 != sizeV2) return 0;
    double result = 0.0;
    for (size_t i = 0; i < sizeV1; ++i) result += v1[i] * v2[i];
    return result;
}

double* getR0(double* b, size_t sizeB, double* A, size_t rowsA, size_t columnsA, double* x, size_t sizeX) {
    double* buffer = multMatrix(A, rowsA, columnsA, x, sizeX, 1);
    double* result = subMatrix(b, sizeB, 1, buffer, sizeX, 1);

    fre(buffer);

    return result;
}

double* getZ0(double* r0, size_t sizeR0) {
    double* result = calloc(sizeR0, sizeof(double));
    if (!result) return NULL;
    for (size_t i = 0; i < sizeR0; ++i) result[i] = r0[i];
    return result;
}

double getNextAlpha(double* rn, size_t sizeRn, double* A, size_t rowsA, size_t columnsA, double* zn, size_t sizeZn) {
    double numerator = nonScalarProduct(rn, sizeRn, rn, sizeRn);

    double* tmp = multMatrix(A, rowsA, columnsA, zn, sizeZn, 1);
    double denominator = nonScalarProduct(tmp, sizeZn, zn, sizeZn);
    free(tmp);

    return numerator / denominator;
}

double* getNextR(double* rn, size_t sizeRn, double* A, size_t rowsA, size_t columnsA, double* zn, size_t sizeZn) {
    if (sizeRn != rowsA || sizeRn != sizeZn) return NULL;
    double nextAlpha = getNextAlpha(rn, sizeRn, A, rowsA, columnsA, zn, sizeZn);
    double* tmp = multMatrix(A, rowsA, columnsA, zn, sizeZn, 1);
    double* buffer = multByValue(nextAlpha, tmp, rowsA, 1);

    double* result = subMatrix(rn, sizeRn, 1, buffer, sizeRn, 1);

    free(tmp);
    free(buffer);

    return result;
}

double getNextBeta(double* nextRn, size_t sizeNextRn, double* rn, size_t sizeRn) {
    if (sizeRn != sizeNextRn) return 0;
    double numerator = nonScalarProduct(nextRn, sizeNextRn, nextRn, sizeNextRn);
    double denominator = nonScalarProduct(rn, sizeRn, rn, sizeRn);
    return numerator / denominator;
}

double* getNextZ(double* nextRn, size_t sizeNextRn, double nextBeta, double* zn, size_t sizeZn) {
    if (sizeNextRn != sizeZn) return NULL;
    double* tmp = multByValue(nextBeta, zn, sizeZn, 1);
    double* result = sumMatrix(nextRn, sizeNextRn, 1, tmp, sizeZn, 1);

    free(tmp);

    return result;
}

double* getNextX(double* xn, size_t sizeXn, double nextAlpha, double* zn, size_t sizeZn) {
    if (sizeXn != sizeZn) return NULL;

    double* tmp = multByValue(nextAlpha, zn, sizeZn, 1);
    double* result = sumMatrix(xn, sizeXn, 1, tmp, sizeXn, 1);

    free(tmp);

    return result;
}

double normaVect(double* u, size_t sizeU) {
    double tmp = nonScalarProduct(u, sizeU, u, sizeU);
    return sqrt(tmp);
}

bool checkingResult(double* r, size_t sizeR, double* b, size_t sizeB) {
    double numerator = normaVect(r, sizeR);
    double denominator = normaVect(b, sizeB);
    double result = numerator / denominator;
    return result < CALCULATIN_GERROR;
}

bool solve(double* A, size_t rowsA, size_t columnsA, double** x, size_t sizeX, double* b, size_t sizeB) {
    double* r = getR0(b, sizeB, A, rowsA, columnsA, *x, sizeX);
    size_t sizeR = sizeB;
    double* z = getZ0(r, sizeR);
    size_t sizeZ = sizeR;

    double nextAlpha = getNextAlpha(r, sizeR, A, rowsA, columnsA, z, sizeZ);
    double* nextX = getNextX(*x, sizeX, nextAlpha, z, sizeZ);
    free(*x);
    *x = nextX;

    double* nextR = NULL;
    double* nextZ = NULL;
    size_t sizeNextR = sizeR;
    double nextBeta = 0;

    for (size_t i = 0; !checkingResult(r, sizeR, b, sizeB); ++i) {
        nextR = getNextR(r, sizeR, A, rowsA, columnsA, z, sizeZ); // r^n+1
        nextBeta = getNextBeta(nextR, sizeNextR, r, sizeR);       // beta^n+1
        nextZ = getNextZ(nextR, sizeNextR, nextBeta, z, sizeZ); // z^n+1
        free(r);
        free(z);
        r = nextR;
        z = nextZ;

        nextAlpha = getNextAlpha(r, sizeR, A, rowsA, columnsA, z, sizeZ); // alpha^n+2
        nextX = getNextX(*x, sizeX, nextAlpha, z, sizeZ);  //x^n+2
        free(*x);
        *x = nextX;
    }

    return true;
}

int main(int argc, char* argv[]) {
    size_t N = 0;
    size_t Nx = N_x;
    size_t Ny = N_y;
    double* A = NULL;
    double* b = NULL;
    if (!getInitData(Nx, Ny, &N, &A, &b)) {
        return 0;
    }

    printf("Array:\n");
    printArray(A, N, N);
    printf("Vector b:\n");
    printArray(b, N, 1);
    double* x = getX0(N);
    printf("Vector x0:\n");
    printArray(x, N, 1);

    solve(A, N, N, &x, N, b, N);
    printf("My result:\n");
    printArray(x, N, 1);
    free(A);
    free(b);
    free(x);
    return 0;
}e
