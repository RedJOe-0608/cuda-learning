#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__global__ void matrixAdd_kernel(const float *a, const float *b, float *c, int num_rows, int num_cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < num_rows && col < num_cols){
        int index = row * num_cols + col;
        c[index] = a[index] + b[index];
    }
}

int main() {
    int num_rows = 1000;
    int num_cols = 500;

    size_t bytes = (num_rows * num_cols) * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    for(int i=0; i< num_rows * num_cols; i++){
        h_a[i] = (float)(i+1);
        h_b[i] = (float)(i+1) * 2;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a,bytes);
    cudaMalloc((void**)&d_b,bytes);
    cudaMalloc((void**)&d_c,bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16);

    dim3 blocksPerGrid(
        ((num_cols + threadsPerBlock.x - 1) / threadsPerBlock.x),
        ((num_rows + threadsPerBlock.y - 1) / threadsPerBlock.y)
    );
    // grid = 32 x 63 blocks, and each block = 16 x 16 threads

    matrixAdd_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, num_rows, num_cols);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);


    int success = 1, num_elements = num_rows * num_cols;
    for(int i=0;i<num_elements;i++){
        if(fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5f){
            printf("Error at index %d: expected %f but got %f", i, (h_a[i] + h_b[i]), h_c[i]);
            success = 0;
        }
    }

    if(success){
        printf("All elements are correct\n");
    }

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
