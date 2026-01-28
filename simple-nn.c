#include <matrix.h>
#include <stdbool.h>
#include <math.h>

typedef struct 
{
    Matrix W;  // weights cache        (out_dim x in_dim)
    Matrix b;  // biases cache         (out_dim x 1)

    Matrix X;  // input cache          (in_dim  x batch)
    Matrix Z;  // pre-activation cache (out_dim x batch)
    Matrix A;  // activations cache    (out_dim x batch) 

    Matrix dW; // backprop weights - same shape as W
    Matrix dB; // backprop biases  - same shape as b

    // dims of layer
    int in_dim;
    int out_dim;

} DenseLayer;

typedef struct 
{
    Matrix A; // cache output of A = sigmoid_forward(Z) for backward prop

} Sigmoid;

typedef struct
{
    Matrix A; // cache output of A = reLU_forward(Z) for backward prop

} ReLU;

void dense_init(DenseLayer* layer, int in_dim, int out_dim);
void dense_forward(DenseLayer* layer, const Matrix* X, Matrix* Z_out, bool training);
void dense_backward(DenseLayer* layer, Matrix* dZ, Matrix* dA_out);
void dense_free(DenseLayer* layer);
void dense_zero_grads(DenseLayer* layer);


void sigmoid_forward(Sigmoid *s, const Matrix* Z, Matrix* A_out, bool training);
void sigmoid_backward(Sigmoid *s, const Matrix* dA, Matrix* dZ_out);


void relu_forward(ReLU* layer, const Matrix* X, Matrix *A_out, bool training);
void relu_backward(ReLU* layer, const Matrix* dA, Matrix* dZ_out);

void dense_forward(DenseLayer* layer, const Matrix* X, Matrix* Z_out, bool training)
{
    if (!X->data || !Z_out->data) return;

    mat_mul(Z_out, &layer->W, X); // Z = W*A 

    for (int i = 0; i < Z_out->rows; i++) 
    {
        float bi = layer->b.data[i]; // b is (out_dim x 1)

        for (int j = 0; j < Z_out->cols; j++) 
        {
            Z_out->data[i * Z_out->cols + j] += bi; // Z = W*A + b
        }
    }

    if (training) // cache params for backprop
    {
        mat_copy((Matrix*)X, &layer->X);
        mat_copy(Z_out, &layer->Z);
    }
}

void dense_backward(DenseLayer* layer, Matrix* dZ, Matrix* dA_out)
{
    if (!layer || !dZ) return;

    // dW = dZ * X^T
    // dZ: (out_dim x batch)
    // X : (in_dim x batch)
    // dW: (out_dim x in_dim)
    mat_mul_A_BT(&layer->dW, dZ, &layer->X);

    // db = sum over columns of dZ
    // db: (out_dim x 1)
    mat_sum_cols(&layer->dB, dZ);

    // dX = W^T * dZ   (only if caller wants it)
    // W : (out_dim x in_dim)
    // dZ: (out_dim x batch)
    // dX: (in_dim x batch)
    if (dA_out) {
        mat_mul_AT_B(dA_out, &layer->W, dZ);
    }

    // Optional: average gradients over batch
    float inv_batch = 1.0f / (float)dZ->cols;
    mat_scale(&layer->dW, inv_batch);
    mat_scale(&layer->dB, inv_batch);
}

void sigmoid_forward(Sigmoid *s, const Matrix* Z, Matrix* A_out, bool training)
{
    size_t n = (size_t)Z->rows * (size_t)Z->cols;

    for (int i = 0; i < n; i++)
    {
        A_out->data[i] = 1.0f / ( 1.0f + expf(-1.0f * Z->data[i]) );
    }

    if (training)
    {
        mat_copy(A_out, &s->A);
    }
}

void relu_forward(ReLU* layer, const Matrix* Z, Matrix *A_out, bool training)
{
    size_t n = (size_t)Z->rows * (size_t)Z->cols;

    for (int i = 0; i < n; i++)
    {
        A_out->data[i] = (Z->data[i] > 0.0f) ? Z->data[i] : 0;
    }

    if (training)
    {
        mat_copy(A_out, &layer->A);
    }

}



//----


typedef struct
{
    int nx; // in_dim
    int nh; // hidden_layer size
    int ny; // out_dim
} LayerDims;


void initialize_params(LayerDims* layer_dim)
{
    
}


