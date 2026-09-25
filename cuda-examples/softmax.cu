#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

float max(const float *arr, int n){
    float max = 0.0f;
    for(int i=0;i<n;i++){
        if(arr[i] > max)
            max = arr[i];
    }

    return max;
}

// So, for every batch, it performs 3 passes: 1, to find the max, 2, to find the exponential sum, and 3, to normalise and write the output.
void softmax_cpu(const float *a, float *b, int num_rows, int num_cols){

    for(int row = 0; row < num_rows; row++){

        // calculate the max for the current row
        float max_value = max(a, num_rows * num_cols);

        float exp_sum = 0.0f;
        for(int col = 0; col < num_cols; col++){
            exp_sum += exp(a[row * num_cols + col] - max_value);
        }

        for(int col = 0; col < num_cols; col++){
            b[row * num_cols + col] = exp(a[row * num_cols + col] - max_value) / exp_sum; 
        }
    }

}

__global__ void softmax_kernel(const float *a, float *b, int num_rows, int num_cols){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < num_rows && col < num_cols){

        float max_value = -1e20f;
        for(int col_idx = 0; col_idx < num_cols; col_idx++){
            if(a[row * num_cols + col_idx] > max_value)
                max_value= a[row * num_cols + col_idx];  
        }

        float sum_exp = 0.0f;
        for(int col_idx = 0; col_idx < num_cols; col_idx++){
                sum_exp += expf(a[row * num_cols + col_idx] - max_value); 
        }

        b[row * num_cols + col] = (a[row * num_cols + col] - max_value) / sum_exp;

    }
}