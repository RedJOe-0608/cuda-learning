#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__global__ void vecAdd_kernel(const float *a, const float *b, float *c, int num_elements) {
    // CUDA gives each thread its coordinates (when we launch the kernel).
    // We use those coordinates to calculate which piece of our data that thread should work on.
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    printf("block=%d, thread=%d, index=%d \n", blockIdx.x,threadIdx.x, index);

    if(index < num_elements) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    int num_elements = 16;
    size_t bytes = num_elements * sizeof(float);

    // allocating memory on the host = CPU
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // initialising the arrays with numbers.
    for(int i=0;i<num_elements;i++){
        h_a[i] = (float)i+1;
        h_b[i] = (float)((i+1)*2);
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a,bytes); // these are essentially CUDA api calls that the CPU makes
    cudaMalloc((void**)&d_b,bytes);
    cudaMalloc((void**)&d_c,bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice); // another CUDA api.
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    vecAdd_kernel<<<1,16>>>(d_a, d_b, d_c, num_elements);
    // vecAdd_kernel<<<2,8>>>(d_a, d_b, d_c, num_elements);
    // vecAdd_kernel<<<4,4>>>(d_a, d_b, d_c, num_elements);
    // both of the above are also valid. But choosing a valid block size has implications on cooperation, performance, scheduling. Do not worry about it yet. 

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    int success = 1;
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
    cudaFree(d_a); // another CUDA api.
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}