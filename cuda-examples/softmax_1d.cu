#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__global__ void softmax_kernel_1d(const float *a, float *b, int num_elements){
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    
    float sum = 0.0f;

    float max = a[0];

    for(int j=0; j < num_elements; j++){
        if(a[j] > max)
            max = a[j];
    }

    // every thread calculates this sum separately. this works, but as you can see, this is scope for improvement.
    // We subtract from max for numerical stability.
    for(int j=0; j < num_elements; j++){
        float exp_j = exp(a[j] - max);
        sum+= exp_j;
    }

    if(i < num_elements){
        printf("Thread ID used: (%d, %d)\n", blockIdx.x, threadIdx.x);
        b[i] = exp(a[i] - max) / sum;
    }
}

int main() {

    int num_elements = 10;

    size_t bytes = num_elements * sizeof(float);

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);

    for(int i=0;i< num_elements;i++){
        h_a[i] = (float)(i+1);
    }

    printf("Original array: \n");
    for(int i=0;i<num_elements;i++){
        printf("%0.0f, ", h_a[i]);
    }
    printf("\n");

    float *d_a, *d_b;

    cudaMalloc((void**)&d_a,bytes);
    cudaMalloc((void**)&d_b,bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 8;
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock; // naive division is num_elements / threadsPerBlock. this will tell us how many blocks do we need. But the above is called ceil division (a + b - 1) / b

    softmax_kernel_1d<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, num_elements);

    cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);

    printf("After softmax: \n");
    for(int i=0;i<num_elements;i++){
        printf("%.6f, ", h_b[i]);
    }

    free(h_a);
    free(h_b);
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}