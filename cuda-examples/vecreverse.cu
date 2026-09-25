#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

__global__ void vec_reverse(const float *a, float *b, int num_elements){
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    
    // blockIdx: Which block am I? 
    // blockDim: How many threads do i contain?
    // threadIdx: Which thread am i within this block?
    // It is much easier to visualise in 1D.
    if(i < num_elements){
        printf("Thread ID used: (%d, %d)\n", blockIdx.x, threadIdx.x);
        b[i] = a[num_elements - i - 1];
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

    vec_reverse<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, num_elements);

    cudaMemcpy(h_b, d_b, bytes, cudaMemcpyDeviceToHost);

    printf("Reversed array: \n");
    for(int i=0;i<num_elements;i++){
        printf("%0.0f, ", h_b[i]);
    }

    free(h_a);
    free(h_b);
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}