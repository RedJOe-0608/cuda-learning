#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

void gemm_cpu(const float *a, const float *b, float *c, int num_rows, int num_cols, int matching_k_dim){
    for(int row = 0; row < num_rows; row++){
        for(int col = 0; col < num_cols; col++){
            float sum = 0.0;
            for(int k=0; k< matching_k_dim; k++){
                sum+= a[row * matching_k_dim + k] * b[k*num_cols + col]; // this is effectively a[row][k] * b[k][col]
            }
            c[row * num_cols + col] = sum;
        }
    }
}

__global__ void gemm_kernel(const float *a, const float *b, float *c, int num_rows, int num_cols, int matching_k_dim){

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < num_rows && col < num_cols){
        float sum = 0.0f;
        for(int k=0;k<matching_k_dim;k++){
            sum+= a[row * matching_k_dim + k] * b[k * num_cols + col];
        }
        c[row * num_cols + col] = sum;
    }
}

int main() {

    // num_rows is number of rows of a, and num_cols is number of rows of b.
    int num_rows = 2, matching_k_dim = 3, num_cols = 2;

    // memory calculation
    size_t bytes_a = (num_rows * matching_k_dim) * sizeof(float);
    size_t bytes_b = (matching_k_dim * num_cols) * sizeof(float);
    size_t bytes_c = (num_rows * num_cols) * sizeof(float);

    // memory allocation
    float *h_a = (float*)malloc(bytes_a);
    float *h_b = (float*)malloc(bytes_b);
    float *h_c = (float*)malloc(bytes_c); // for cpu
    float *h_d = (float*)malloc(bytes_c); // for gpu

    // memory initialisation
    for(int i=0;i< num_rows * matching_k_dim; i++){
        h_a[i] = (float)(i+1);
    }

    for(int i=0;i< matching_k_dim * num_cols; i++){
        h_b[i] = (float)(i+1) * 2;
    }

    gemm_cpu(h_a, h_b, h_c, num_rows, num_cols, matching_k_dim);

    // printing the output.
    for(int row = 0; row < num_rows; row++){
        for(int col = 0; col < num_cols; col++){
            printf("%f ", h_c[row * num_cols + col]);
        }
        printf("\n");
    }

    float *d_a, *d_b, *d_c;

    cudaMalloc((void**)&d_a,bytes_a);
    cudaMalloc((void**)&d_b,bytes_b);
    cudaMalloc((void**)&d_c,bytes_c);

    cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16,16);

    dim3 blocksPerGrid(
        ((num_cols + threadsPerBlock.x - 1) / threadsPerBlock.x),
        ((num_rows + threadsPerBlock.y -1) / threadsPerBlock.y)
    );

    gemm_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, num_rows, num_cols, matching_k_dim);

    cudaMemcpy(h_d, d_c, bytes_c, cudaMemcpyDeviceToHost);

    // gpu
    printf("\n");
    printf("---GPU kernel output---\n");
    for(int row = 0; row < num_rows; row++){
        for(int col = 0; col < num_cols; col++){
            printf("%f ", h_d[row * num_cols + col]);
        }
        printf("\n");
    }

    return 0;
}