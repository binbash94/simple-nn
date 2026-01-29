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

typedef struct {
    DenseLayer fc1;
    ReLU  relu1;
    DenseLayer fc2;
    ReLU  relu2;
    DenseLayer fc3;     // output dim = 1

    Sigmoid sigmoid;    // sigmoid + BCE

    // Scratch buffers
    Matrix z1, a1;
    Matrix z2, a2;
    Matrix logits;      // (1 x batch)
    Matrix y_hat;       // (1 x batch)

    Matrix dlogits;     // (1 x batch)
    Matrix da2, dz2;
    Matrix da1, dz1;

    int input_dim;
    int hidden1;
    int hidden2;
    int max_batch;
} MLP;


typedef struct 
{
    Matrix A; // cache output of A = sigmoid_forward(Z) for backward prop

} Sigmoid;

typedef struct
{
    Matrix A; // cache output of A = reLU_forward(Z) for backward prop

} ReLU;

void dense_init(DenseLayer *l, int in_dim, int out_dim, int max_batch);
void dense_forward(DenseLayer* layer, const Matrix* X, Matrix* Z_out, bool training);
void dense_backward(DenseLayer *l, const Matrix *dZ, Matrix *dA_out);
void dense_free(DenseLayer* layer);
void dense_zero_grads(DenseLayer* layer);


void sigmoid_forward(Sigmoid *layer, const Matrix* Z, Matrix* A_out, bool training);
void sigmoid_backward(Sigmoid *layer, const Matrix* dA, Matrix* dZ_out);


void relu_init(ReLU *r, int rows, int max_batch);
void relu_forward(ReLU* layer, const Matrix* Z, Matrix *A_out, bool training);
void relu_backward(ReLU* layer, const Matrix* dA, Matrix* dZ_out);

float binary_cross_entropy(const Matrix *A, const Matrix *Y);
void binary_cross_entropy_backward(const Matrix *A, const Matrix *Y, Matrix *dZ_out);

bool mlp_init(MLP *m, int input_dim, int hidden1, int hidden2, int max_batch);



void dense_init(DenseLayer *l, int in_dim, int out_dim, int max_batch)
{
    l->in_dim  = in_dim;
    l->out_dim = out_dim;

    mat_alloc(&l->W,  out_dim, in_dim);
    mat_alloc(&l->b,  out_dim, 1);
    mat_alloc(&l->X,  in_dim,  max_batch);
    mat_alloc(&l->Z,  out_dim, max_batch);
    mat_alloc(&l->A,  out_dim, max_batch);
    mat_alloc(&l->dW, out_dim, in_dim);
    mat_alloc(&l->dB, out_dim, 1);

    mat_rand_uniform(&l->W, -0.01f, 0.01f);
    mat_zero(&l->b);
    mat_zero(&l->dW);
    mat_zero(&l->dB);
}

void dense_zero_grads(DenseLayer *l)
{
    mat_zero(&l->dW);
    mat_zero(&l->dB);
}

void dense_forward(DenseLayer *l, const Matrix *X, Matrix *Z_out, bool training)
{
    // Z = W·X + b
    mat_mul(Z_out, &l->W, X);
    mat_add_bias_cols(Z_out, &l->b);

    if (training) {
        mat_copy(&l->X, X);
        mat_copy(&l->Z, Z_out);
    }
}

void dense_backward(DenseLayer *l, const Matrix *dZ, Matrix *dA_out)
{
    int batch = dZ->cols;

    // dW = dZ · X^T
    mat_mul_A_BT(&l->dW, dZ, &l->X);

    // dB = sum_cols(dZ)
    mat_sum_cols(&l->dB, dZ);

    float invB = 1.0f / (float)batch;
    mat_scale(&l->dW, invB);
    mat_scale(&l->dB, invB);

    // dA = W^T · dZ
    if (dA_out) {
        mat_mul_AT_B(dA_out, &l->W, dZ);
    }
}

void dense_free(DenseLayer *l)
{
    mat_free(&l->W);
    mat_free(&l->b);
    mat_free(&l->X);
    mat_free(&l->Z);
    mat_free(&l->A);
    mat_free(&l->dW);
    mat_free(&l->dB);
}

/* =========================
   ReLU
   ========================= */

typedef struct
{
    Matrix Z;   // cache pre-activation
} ReLU;

void relu_init(ReLU *r, int rows, int max_batch)
{
    mat_alloc(&r->Z, rows, max_batch);
}

void relu_forward(ReLU *r, const Matrix *Z, Matrix *A_out, bool training)
{
    size_t n = (size_t)Z->rows * Z->cols;

    for (size_t i = 0; i < n; ++i) {
        float z = Z->data[i];
        A_out->data[i] = (z > 0.0f) ? z : 0.0f;
    }

    if (training) {
        mat_copy(&r->Z, Z);
    }
}

void relu_backward(ReLU *r, const Matrix *dA, Matrix *dZ_out)
{
    size_t n = (size_t)dA->rows * dA->cols;

    for (size_t i = 0; i < n; ++i) {
        dZ_out->data[i] =
            (r->Z.data[i] > 0.0f) ? dA->data[i] : 0.0f;
    }
}

void relu_free(ReLU *r)
{
    mat_free(&r->Z);
}

/* =========================
   Sigmoid
   ========================= */

typedef struct
{
    Matrix A;   // cache output
} Sigmoid;

void sigmoid_init(Sigmoid *s, int rows, int max_batch)
{
    mat_alloc(&s->A, rows, max_batch);
}

void sigmoid_forward(Sigmoid *s, const Matrix *Z, Matrix *A_out, bool training)
{
    size_t n = (size_t)Z->rows * Z->cols;

    for (size_t i = 0; i < n; ++i) {
        A_out->data[i] = 1.0f / (1.0f + expf(-Z->data[i]));
    }

    if (training) {
        mat_copy(&s->A, A_out);
    }
}

void sigmoid_backward(Sigmoid *s, const Matrix *dA, Matrix *dZ_out)
{
    size_t n = (size_t)dA->rows * dA->cols;

    for (size_t i = 0; i < n; ++i) {
        float a = s->A.data[i];
        dZ_out->data[i] = dA->data[i] * a * (1.0f - a);
    }
}

void sigmoid_free(Sigmoid *s)
{
    mat_free(&s->A);
}

/* =========================
   Binary Cross-Entropy Loss
   ========================= */

float binary_cross_entropy(const Matrix *A, const Matrix *Y)
{
    float loss = 0.0f;
    int B = A->cols;

    for (int i = 0; i < B; ++i) {
        float a = A->data[i];
        float y = Y->data[i];
        loss += -(y * logf(a + 1e-8f) + (1.0f - y) * logf(1.0f - a + 1e-8f));
    }

    return loss / (float)B;
}

// For sigmoid + BCE: dZ = A - Y
void binary_cross_entropy_backward(const Matrix *A, const Matrix *Y, Matrix *dZ_out)
{
    size_t n = (size_t)A->rows * A->cols;

    for (size_t i = 0; i < n; ++i) {
        dZ_out->data[i] = A->data[i] - Y->data[i];
    }
}

bool mlp_init(MLP *m,
              int input_dim,
              int hidden1,
              int hidden2,
              int max_batch)
{
    if (!m) return false;

    memset(m, 0, sizeof(*m));

    m->input_dim = input_dim;
    m->hidden1   = hidden1;
    m->hidden2   = hidden2;
    m->max_batch = max_batch;

    /* ---------- Dense layers ---------- */
    dense_init(&m->fc1, input_dim, hidden1, max_batch);
    dense_init(&m->fc2, hidden1, hidden2, max_batch);
    dense_init(&m->fc3, hidden2, 1,        max_batch);

    /* ---------- Activations ---------- */
    relu_init(&m->relu1, hidden1, max_batch);
    relu_init(&m->relu2, hidden2, max_batch);

    // Sigmoid cache for output layer
    mat_alloc(&m->sigmoid.A, 1, max_batch);

    /* ---------- Forward scratch buffers ---------- */
    mat_alloc(&m->z1, hidden1, max_batch);
    mat_alloc(&m->a1, hidden1, max_batch);

    mat_alloc(&m->z2, hidden2, max_batch);
    mat_alloc(&m->a2, hidden2, max_batch);

    mat_alloc(&m->logits, 1, max_batch);
    mat_alloc(&m->y_hat,  1, max_batch);

    /* ---------- Backward scratch buffers ---------- */
    mat_alloc(&m->dlogits, 1, max_batch);
    mat_alloc(&m->da2, hidden2, max_batch);
    mat_alloc(&m->dz2, hidden2, max_batch);
    mat_alloc(&m->da1, hidden1, max_batch);
    mat_alloc(&m->dz1, hidden1, max_batch);

    return true;
}


float mlp_train_step(MLP *m,
                     const Matrix *X,   // (input_dim x batch)
                     const Matrix *y,   // (1 x batch), values 0 or 1
                     float lr)
{
    /* =====================
       Forward pass
       ===================== */

    // Layer 1
    dense_forward(&m->fc1, X, &m->z1, true);
    relu_forward(&m->relu1, &m->z1, &m->a1, true);

    // Layer 2
    dense_forward(&m->fc2, &m->a1, &m->z2, true);
    relu_forward(&m->relu2, &m->z2, &m->a2, true);

    // Output layer (logits)
    dense_forward(&m->fc3, &m->a2, &m->logits, true);

    // Sigmoid + BCE
    sigmoid_forward(&m->sigmoid, &m->logits, &m->y_hat, true);
    float loss = binary_cross_entropy(&m->y_hat, y);

    /* =====================
       Backward pass
       ===================== */

    // dZ3 = y_hat - y
    binary_cross_entropy_backward(&m->y_hat, y, &m->dlogits);

    // Layer 3
    dense_backward(&m->fc3, &m->dlogits, &m->da2);
    relu_backward(&m->relu2, &m->da2, &m->dz2);

    // Layer 2
    dense_backward(&m->fc2, &m->dz2, &m->da1);
    relu_backward(&m->relu1, &m->da1, &m->dz1);

    // Layer 1
    dense_backward(&m->fc1, &m->dz1, NULL);

    /* =====================
       SGD update (ONCE)
       ===================== */

    // W := W - lr * dW, b := b - lr * dB
    mat_scale(&m->fc1.dW, lr); mat_sub(&m->fc1.W, &m->fc1.dW);
    mat_scale(&m->fc2.dW, lr); mat_sub(&m->fc2.W, &m->fc2.dW);
    mat_scale(&m->fc3.dW, lr); mat_sub(&m->fc3.W, &m->fc3.dW);

    mat_scale(&m->fc1.dB, lr); mat_sub(&m->fc1.b, &m->fc1.dB);
    mat_scale(&m->fc2.dB, lr); mat_sub(&m->fc2.b, &m->fc2.dB);
    mat_scale(&m->fc3.dB, lr); mat_sub(&m->fc3.b, &m->fc3.dB);

    mat_zero(&m->fc1.dW);
    mat_zero(&m->fc1.dB);
    
    mat_zero(&m->fc2.dW);
    mat_zero(&m->fc2.dB);
    
    mat_zero(&m->fc3.dW);
    mat_zero(&m->fc3.dB);


    return loss;
}

void mlp_train(MLP *m,
               Dataset *data,
               int epochs,
               float lr)
{
    for (int e = 0; e < epochs; ++e) {
        float epoch_loss = 0.0f;

        for (int i = 0; i < data->num_batches; ++i) {
            epoch_loss += mlp_train_step(
                m,
                &data->X_batches[i],
                &data->Y_batches[i],
                lr
            );
        }

        printf("epoch %d | loss %.4f\n", e, epoch_loss / data->num_batches);
    }
}
