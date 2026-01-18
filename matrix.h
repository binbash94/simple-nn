// nn-matrix.c - minimal matrix library implementation
#include <stdio.h>
#include <libc.h>

typedef struct {
	int rows;
	int cols;
	float *data; // row-major, contiguous: data[r*cols + c]
} Matrix;


void mat_zero(Matrix *m); //sets all elements in matrix to 0
void mat_fill(Matrix *m, float v); //fills all elements of the matrix with value v 
void mat_copy(Matrix *first, Matrix *second); // copys matrix from first to second
void mat_free(Matrix *m); // free allocated memory for matrix m
void mat_size(const Matrix *m); // return size of matrix


Matrix* mat_mul(Matrix *product, const Matrix *first, const Matrix *second); // dot product of two matricies.
Matrix* mat_add(Matrix *product, const Matrix *first, const Matrix *second); // adds two matricies
Matrix* mat_div(Matrix *m, float scalar); // divides each element by scalar