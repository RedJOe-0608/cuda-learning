#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

    void matrix_transpose_cpu(const float *a, float *b, int num_rows, int num_cols){
        for(int row=0; row < num_rows; row++){
            for(int col = 0; col < num_cols; col++){
                  b[col*num_rows + row] = a[row*num_cols + col];
            }
        }

         // Print b as a matrix
        for (int row = 0; row < num_cols; row++) {
            for (int col = 0; col < num_rows; col++) {
                printf("%f ", b[row * num_rows + col]);
            }
            printf("\n");
        }
    }

__global__ void matrix_transpose_kernel(const float *a, float *b, int num_rows, int num_cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < num_rows && col < num_cols){
        b[col*num_rows + row] = a[row*num_cols + col];
    }
}

int main() {

    int num_rows = 2, num_cols = 3;

    size_t bytes = (num_rows * num_cols) * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes); //for cpu output

    for(int i=0; i< num_rows * num_cols; i++){
        h_a[i] = float(i+1);
    }

    matrix_transpose_cpu(h_a, h_c, num_rows, num_cols);

    float *d_a, *d_b;

    cudaMalloc((void**)&d_a,bytes);
    cudaMalloc((void**)&d_b,bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16);

    dim3 blocksPerGrid(
        ((num_cols + threadsPerBlock.x - 1) / threadsPerBlock.x),
        ((num_rows + threadsPerBlock.y - 1) / threadsPerBlock.y)
    );

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, num_rows, num_cols);

    cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);

    // GPU kernel output.
    printf("\n");
    printf("---GPU OUTPUT---\n");
    for(int row=0; row < num_cols; row++){
        for(int col = 0; col < num_rows; col++){
            printf("%f ", h_b[row * num_rows + col]);
        }
        printf("\n");
    }

    free(h_a);
    free(h_b);
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}
