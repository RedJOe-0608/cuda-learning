#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

void conv_1d_cpu(const float *a, const float *kernel, float *c, int input_size, int kernel_size){
    int output_size = input_size - kernel_size + 1;

    for(int i=0; i<output_size;i++){
        float result = 0.0f;
        for(int j = 0; j < kernel_size; j++){
            result += a[i + j] * kernel[j];
        }
        c[i] = result;
    }

    for(int i=0;i<output_size;i++){
        printf("%0.0f ", c[i]);
    }
}

__global__ void conv_1d_kernel(const float *a, const float *kernel, float *c, int input_size, int kernel_size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int output_size = input_size - kernel_size + 1;

    if(i < output_size){
        float sum = 0.0f;
        for(int j = 0; j < kernel_size; j++){
            sum+= a[i + j] * kernel[j];
        }
        c[i] = sum;
    }
}

int main(){

    int input_size = 100000, kernel_size = 32;
    int output_size =(input_size - kernel_size) + 1;

    size_t input_bytes = input_size * sizeof(float);
    size_t kernel_bytes = kernel_size * sizeof(float);
    size_t output_bytes = output_size * sizeof(float);

    float *h_a = (float*)malloc(input_bytes);
    float *h_kernel = (float*)malloc(kernel_bytes);
    float *h_c = (float*)malloc(output_bytes);

    for(int i=0;i<input_size;i++){
        h_a[i] = (float)(i+1);

        if(i < kernel_size){
            h_kernel[i] = (float)(10 * (i+1));
        }
    }

    conv_1d_cpu(h_a, h_kernel, h_c, input_size, kernel_size);

    float *d_a, *d_kernel, *d_c;

    cudaMalloc((void**)&d_a,input_bytes);
    cudaMalloc((void**)&d_kernel,kernel_bytes);
    cudaMalloc((void**)&d_c,output_bytes);

    cudaMemcpy(d_a, h_a, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 8;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    conv_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_kernel, d_c, input_size, kernel_size);

    cudaMemcpy(h_c, d_c, output_bytes, cudaMemcpyDeviceToHost);

    free(h_a);
    free(h_kernel);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_kernel);
    cudaFree(d_c);

    return 0;
}