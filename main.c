#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <matrix.h>
#include <simple-nn.c>

static uint32_t read_be_u32(FILE *f)
{
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) {
        return 0;
    }

    return ((uint32_t)b[0] << 24) |
           ((uint32_t)b[1] << 16) |
           ((uint32_t)b[2] << 8) |
           (uint32_t)b[3];
}

Matrix load_mnist_images_idx(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open image file: %s\n", path);
        exit(1);
    }

    uint32_t magic = read_be_u32(f);
    uint32_t count = read_be_u32(f);
    uint32_t rows = read_be_u32(f);
    uint32_t cols = read_be_u32(f);

    if (magic != 2051) {
        fprintf(stderr, "Invalid MNIST image magic number in %s\n", path);
        exit(1);
    }

    size_t image_size = (size_t)rows * (size_t)cols;
    unsigned char *raw = (unsigned char *)malloc((size_t)count * image_size);
    if (!raw) {
        fprintf(stderr, "Failed to allocate image buffer\n");
        exit(1);
    }

    size_t expected = (size_t)count * image_size;
    if (fread(raw, 1, expected, f) != expected) {
        fprintf(stderr, "Failed to read all MNIST image bytes from %s\n", path);
        exit(1);
    }

    Matrix X = {0};
    mat_alloc(&X, (int)image_size, (int)count);

    for (uint32_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < image_size; ++j) {
            X.data[j + (size_t)i * image_size] = raw[(size_t)i * image_size + j] / 255.0f;
        }
    }

    free(raw);
    fclose(f);

    return X;
}

Matrix load_mnist_labels_idx(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open label file: %s\n", path);
        exit(1);
    }

    uint32_t magic = read_be_u32(f);
    uint32_t count = read_be_u32(f);

    if (magic != 2049) {
        fprintf(stderr, "Invalid MNIST label magic number in %s\n", path);
        exit(1);
    }

    unsigned char *raw = (unsigned char *)malloc(count);
    if (!raw) {
        fprintf(stderr, "Failed to allocate label buffer\n");
        exit(1);
    }

    if (fread(raw, 1, count, f) != count) {
        fprintf(stderr, "Failed to read all MNIST label bytes from %s\n", path);
        exit(1);
    }

    Matrix Y = {0};
    mat_alloc(&Y, 1, (int)count);

    for (uint32_t i = 0; i < count; ++i) {
        Y.data[i] = (float)raw[i];
    }

    free(raw);
    fclose(f);

    return Y;
}

Dataset load_mnist_dataset(const char *images_path, const char *labels_path)
{
    Dataset d = {0};
    d.X_batches = malloc(sizeof(Matrix));
    d.Y_batches = malloc(sizeof(Matrix));
    d.X_batches[0] = load_mnist_images_idx(images_path);
    d.Y_batches[0] = load_mnist_labels_idx(labels_path);
    d.num_batches = 1;

    if (d.X_batches[0].cols != d.Y_batches[0].cols) {
        fprintf(stderr, "MNIST image/label sample counts do not match\n");
        exit(1);
    }

    return d;
}

int main(int argc, char **argv)
{
    const char *images_path = "../archive/train-images.idx3-ubyte";
    const char *labels_path = "../archive/train-labels.idx1-ubyte";

    if (argc >= 3) {
        images_path = argv[1];
        labels_path = argv[2];
    }

    Dataset train = load_mnist_dataset(images_path, labels_path);

    MLP mlp;
    mlp_init(&mlp, train.X_batches[0].rows, 128, 64, 10, train.X_batches[0].cols);

    printf("X shape: rows = %d, cols = %d\n", train.X_batches[0].rows, train.X_batches[0].cols);
    printf("Y shape: rows = %d, cols = %d\n", train.Y_batches[0].rows, train.Y_batches[0].cols);

    mlp_train(&mlp, &train, 40, 0.1f);

    return 0;
}
