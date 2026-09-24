#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

void conv_2d_cpu(const float *a, const float *kernel, float *out, int height, int width, int kernel_size){
    int output_h = height - kernel_size + 1;
    int output_w = width - kernel_size + 1;

    for(int r = 0; r < output_h; r++){
        for(int c = 0; c < output_w; c++){
            float sum = 0.0f;
            for(int kr = 0; kr < kernel_size; kr++){
                for(int kc = 0; kc < kernel_size; kc++){
                    int input_row = r + kr;
                    int input_col = c + kc;

                    sum += a[input_row * width + input_col] * kernel[kr * kernel_size + kc];
                }
            }
            out[r * output_w + c] = sum;
        }
    }

}

__global__ void conv_2d_kernel(const float* in, const float *kernel, float *out, int height, int width, int kernel_size){
    int output_h = height - kernel_size + 1;
    int output_w = width - kernel_size + 1;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < output_h && col < output_w){
        
        float sum = 0.0f;
        for(int kr = 0; kr < kernel_size; kr++){
            for(int kc = 0; kc < kernel_size; kc++){
                int input_row = row + kr;
                int input_col = col + kc;

                sum += in[input_row * width + input_col] * kernel[kr * kernel_size + kc];
            }
        }

        out[row * output_w + col] = sum;
    }
}