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

void mat_copy(Matrix *dst, const Matrix *src)
{
    if (!dst || !src || !dst->data || !src->data) return;
    if (dst->rows != src->rows || dst->cols != src->cols) return;

    memcpy(dst->data, src->data, (size_t)src->rows * (size_t)src->cols * sizeof(float));
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

void mat_scale(Matrix* m, float scalar)
{
	size_t n = (size_t)m->rows * (size_t)m->cols;

	for (int i = 0; i < n; i ++)
	{
		m->data[i] *= scalar;
	}
}

void mat_sub(Matrix *A, const Matrix *B)
{
    if (!A || !B || !A->data || !B->data) return;

    // Shapes must match
    if (A->rows != B->rows || A->cols != B->cols) return;

    size_t n = (size_t)A->rows * (size_t)A->cols;

    for (size_t i = 0; i < n; ++i) {
        A->data[i] -= B->data[i];
    }
}

void mat_sum_cols(Matrix* dst, const Matrix* src)
{
	if (!dst->data || !src->data) return;

	for(int i = 0; i < dst->rows; i++)
	{
		float sum = 0.0f;

		for(int j = 0; j < src->cols; j++)
		{
			sum += src->data[i*src->cols + j];
		}
		
		dst->data[i] = sum;
	}
}

Matrix* mat_mul_AT_B(Matrix *product, const Matrix *first, const Matrix *second)
{
    if (!product || !first || !second) return NULL;
    if (!product->data || !first->data || !second->data) return NULL;

    // first:  (m × n)
    // second: (m × p)
    // product:(n × p)
    if (first->rows != second->rows) return NULL;
    if (product->rows != first->cols) return NULL;
    if (product->cols != second->cols) return NULL;

    int m = first->rows;
    int n = first->cols;
    int p = second->cols;

    for (int i = 0; i < n; i++) {          // rows of first
        for (int j = 0; j < p; j++) {      // cols of second

            float sum = 0.0f;

            for (int k = 0; k < m; k++) {  // rows of first or second
                sum += first->data[k * n + i] * second->data[k * p + j]; 
            }

            product->data[i * p + j] = sum;
        }
    }

    return product;
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

void mat_mul_A_BT(Matrix *C, const Matrix *A, const Matrix *B)
{
    if (!A || !B || !C) return;
    if (!A->data || !B->data || !C->data) return;

    // A: (m x n)
    // B: (p x n)
    // C: (m x p)
    if (A->cols != B->cols) return;
    if (C->rows != A->rows || C->cols != B->rows) return;

    int m = A->rows;
    int n = A->cols;
    int p = B->rows;

    // Compute C[i,j] = sum_k A[i,k] * B[j,k]
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {

            float sum = 0.0f;

            // Inner dot product between:
            // row i of A  and  row j of B
            for (int k = 0; k < n; ++k) {
                sum += A->data[i * n + k] * B->data[j * n + k];
            }

            C->data[i * p + j] = sum;
        }
    }
}

bool mat_alloc(Matrix *m, int r, int c) 
{
    m->rows = r;
    m->cols = c;
    m->data = (float*)malloc((size_t)r * c * sizeof(float));
    return m->data != NULL;
}

void mat_rand_uniform(Matrix *m, float min, float max)
{
	size_t n = (size_t)m->rows * (size_t)m->cols;

	for(size_t i = 0; i < n; i++)
	{
		// (float)rand() / (float)RAND_MAX) to normalize between 0 and 1
		m->data[i] = ((float)rand() / (float)RAND_MAX) * (max - min) + min;
	}
}