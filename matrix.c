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
Matrix* mat_sub(Matrix *product, const Matrix *first, const Matrix *second); // subtracts to matricies
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

void mat_copy(Matrix *first, Matrix *second)
{
	if (!first || !second || !first->data || !second->data) return; 
	if (first->rows != second->rows || first->cols != second->cols) return; 

	memcpy(second->data, first->data, (size_t)second->rows * (size_t)second->cols * sizeof(float));
}

void mat_free(Matrix *m)
{
	if(!m || !m->data) return;
	free(m->data);
}

void mat_size(const Matrix *m)
{
	if(!m) return;
	printf("size: %d x %d", m->rows, m->cols);
}

Matrix* mat_mul(Matrix *product, const Matrix *first, const Matrix *second)
{
	// TODO: use optimizations like BLAS SIMD
	if(first->cols != second->rows) return NULL;
	if( (product->rows != first->rows) && (product->cols != second->cols)  ) return NULL;

    for (int i = 0; i < first->rows; i++) {
        for (int j = 0; j < second->cols; j++) {

            float sum = 0.0f;

            for (int k = 0; k < first->cols; k++) {
                sum += first->data[i * first->cols + k]
                     * second->data[k * second->cols + j];
            }

            product->data[i * second->cols + j] = sum;
        }
    }

    return product;
}

Matrix* mat_add(Matrix *product, const Matrix *first, const Matrix *second)
{
	// TODO: use optimizations like BLAS SIMD
	if (!first || !second || !first->data || !second->data) return NULL; 
	if (product->rows != first->rows || product->cols != first->cols) return NULL;
	if (product->rows != second->rows || product->cols != second->cols) return NULL;

	size_t n = first->rows * first->cols;

	for(int i = 0; i < n ; i++)
	{
		product->data[i] = first->data[i] + second->data[i];
	}

	return product;
}
