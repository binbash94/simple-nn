// nn-matrix.c - minimal matrix library implementation
#include <matrix.h>

void mat_zero(Matrix *m)
{
	if (!m || !m->data) return;
	memset(m->data, 0, (size_t)m->rows * (size_t)m->cols * sizeof(float));
}

void mat_fill(Matrix *m, float v)
{
	if (!m || !m->data) return;
	size_t n = (size_t)m->rows * (size_t)m->cols; // m is a 1D flat array of rows and cols in memory
	for(size_t i = 0; i < n; i++)
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
                sum += first->data[i * first->cols + k] * second->data[k * second->cols + j];
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

	size_t n = (size_t)first->rows * (size_t)first->cols;

	for(int i = 0; i < n ; i++)
	{
		product->data[i] = first->data[i] + second->data[i];
	}

	return product;
}

Matrix* mat_div(Matrix *m, float scalar)
{
	// TODO: use optimizations like BLAS SIMD
	if(!m || !m->data) return NULL;
	if(scalar == 0.0f) return NULL;

	size_t n = (size_t)m->rows * (size_t)m->cols;

	for(size_t i = 0; i < n; i++)
	{
		m->data[i] /= scalar;
	}

	return m;
}