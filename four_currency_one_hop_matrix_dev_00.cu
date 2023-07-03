#include <cuda_runtime.h>
#include <stdio.h>

#define SIZE 4

__global__ void generate_fx_matrix(const int* input, int* output)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int digit1 = tid / (SIZE * SIZE);
    int digit2 = (tid / SIZE) % SIZE;
    int digit3 = tid % SIZE;

    int path_weight = input[digit1] * 100 + input[digit2] * 10 + input[digit3];
    output[tid] = path_weight;

}

__global__ void generate_base_matrix(const int* input, int* output)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int row     = tid / SIZE;
    int digit1  = row / SIZE;
    int digit2  = row % SIZE;

    int path_weight = input[digit1] * 10 + input[digit2];
    output[tid] = path_weight;
}


void visual_matrix(int* output, int siz) 
{

    for (int i = 0; i < siz * siz * siz; i++)
    {
        printf("%d ", output[i]);
        if ((i + 1) % siz == 0)
        {
            printf("\n");
        }

        if ((i + 1) % (siz * siz) == 0)
        {
            printf("\n");
        }
    }
}

int main()
{
    int input[SIZE] = { 1, 2, 3, 4 };
    int output[SIZE * SIZE * SIZE];

    int* d_input;
    int* d_output;

    cudaMalloc((void**)&d_input, sizeof(int) * SIZE);
    cudaMalloc((void**)&d_output, sizeof(int) * SIZE * SIZE * SIZE);

    cudaMemcpy(d_input, input, sizeof(int) * SIZE, cudaMemcpyHostToDevice);

    int numThreads = SIZE * SIZE * SIZE;
    int numBlocks = (numThreads + 255) / 256;

    generate_fx_matrix<<<numBlocks, 256>>>(d_input, d_output);

    cudaMemcpy(output, d_output, sizeof(int) * SIZE * SIZE * SIZE, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

  // Print the path fx weights
    visual_matrix(output, SIZE);

    cudaMalloc((void**)&d_input, sizeof(int) * SIZE);
    cudaMalloc((void**)&d_output, sizeof(int) * SIZE * SIZE * SIZE);

    cudaMemcpy(d_input, input, sizeof(int) * SIZE, cudaMemcpyHostToDevice);

    generate_base_matrix<<<numBlocks, 256>>>(d_input, d_output);

    cudaMemcpy(output, d_output, sizeof(int) * SIZE * SIZE * SIZE, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    // Print the base fx weights
    visual_matrix(output, SIZE);

    return 0;
}
