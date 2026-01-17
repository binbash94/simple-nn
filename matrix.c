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
void mat_copy(Matrix *src, Matrix *dst); // copys matrix from src to dst

char mat_size(const Matrix *m); // return size of matrix


Matrix* mat_mul(const Matrix *first, const Matrix *second); // dot product of two matricies.
Matrix* mat_add(const Matrix *first, const Matrix *second); // adds two matricies
Matrix* mat_sub(const Matrix *first, const Matrix *second); // subtracts to matricies
Matrix* mat_div(const Matrix *m, float num_test_examples); // divides each element by num_test_samples


int main()
{
	return 0;
}

void mat_zero(Matrix *m)
{
	if (!m || !m->data) return;
	memset(m->data, 0, (size_t)m->rows * (size_t)m->cols * sizeof(float));
}

void mat_fill(Matrix *m, float v)
{
	if (!m || !m->data) return;
	size_t n = (size_t)m->rows * (size_t)m->cols; // m is a 1D flat array of rows and cols in memory
	for(int i = 0; i < n; i++)
	{
		m->data[i] = v;
	}
}

void mat_copy(Matrix *src, Matrix *dst)
{
	if (!src || !dst || !src->data || !dst->data) return;
	if (src->rows != dst->rows || src->cols != dst->cols) return; 

	memcpy(dst->data, src->data, (size_t)dst->rows * (size_t)dst->cols * sizeof(float));
}



