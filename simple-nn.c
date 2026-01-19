#include <matrix.h>
#include <stdbool.h>

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

void dense_init(DenseLayer* layer, int in_dim, int out_dim);
void dense_forward(DenseLayer* layer, const Matrix* X, Matrix* Z_out, bool training);
void dense_backward(DenseLayer* layer, Matrix* dZ, Matrix* dX_out);
void dense_free(DenseLayer* layer);
void dense_zero_grads(DenseLayer* layer);

typedef struct 
{
} Sigmoid;

typedef struct
{
} ReLU;



