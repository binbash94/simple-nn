#include <matrix.h>
#include <simple-nn.c>

#include "hdf5.h"

Matrix load_labels_h5(hid_t file, const char *dataset_name)
{
    hid_t dset = H5Dopen(file, dataset_name, H5P_DEFAULT);
    hid_t space = H5Dget_space(dset);

    hsize_t dims[1];
    H5Sget_simple_extent_dims(space, dims, NULL);
    size_t m = dims[0];

    unsigned char *tmp = malloc(m);
    H5Dread(dset, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL,
            H5P_DEFAULT, tmp);

    Matrix Y = {0};

    mat_alloc(&Y, 1, m);

    for (size_t i = 0; i < m; i++)
        Y.data[i] = (float)tmp[i];

    free(tmp);
    H5Sclose(space);
    H5Dclose(dset);

    return Y;
}

Matrix load_images_h5(hid_t file, const char *dataset_name)
{
    hid_t dset = H5Dopen(file, dataset_name, H5P_DEFAULT);
    hid_t space = H5Dget_space(dset);

    hsize_t dims[4];
    H5Sget_simple_extent_dims(space, dims, NULL);

    size_t m = dims[0];
    size_t H = dims[1]; 
    size_t W = dims[2];
    size_t C = dims[3];

    size_t img_size = H * W * C;

    unsigned char *raw = malloc(m * img_size);

    H5Dread(dset, H5T_NATIVE_UCHAR,
            H5S_ALL, H5S_ALL,
            H5P_DEFAULT, raw);

    Matrix X = {0};

    mat_alloc(&X, img_size, m);

    // NHWC → flattened column-major
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < img_size; j++) {
            X.data[j + i * img_size] = raw[i * img_size + j] / 255.0f;
        }
    }

    free(raw);
    H5Sclose(space);
    H5Dclose(dset);

    return X;
}

Dataset load_dataset(const char *path,
                     const char *x_name,
                     const char *y_name)
{
    hid_t file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        printf("Failed to open %s\n", path);
        exit(1);
    }

    Dataset d;
    d.X_batches = malloc(sizeof(Matrix));
    d.Y_batches = malloc(sizeof(Matrix));
    d.X_batches[0] = load_images_h5(file, x_name);
    d.Y_batches[0] = load_labels_h5(file, y_name);
    d.num_batches = 1;

    H5Fclose(file);
    return d;
}


int main()
{
    Dataset train = load_dataset("../archive/train_catvsnoncat.h5", "train_set_x", "train_set_y");

    MLP mlp;

    mlp_init(&mlp, train.X_batches->rows, 512, 128, 209);

    printf("X shape: rows = %d, cols = %d\n",
       train.X_batches[0].rows,
       train.X_batches[0].cols);

    printf("Y shape: rows = %d, cols = %d\n",
       train.Y_batches[0].rows,
       train.Y_batches[0].cols);

    mlp_train(&mlp, &train, 10, 0.01f);

    return 0;
}