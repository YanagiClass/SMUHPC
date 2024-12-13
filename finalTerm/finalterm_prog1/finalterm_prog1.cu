#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DS_timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOCK_SIZE 16

// problem 1

int main(void)
{
    int m = 1024;
    int n = 1024;
    int k = 1024;
    int sizeA = m * k;
    int sizeB = k * n;
    int sizeC = m * n;
    int result;
    float *A, *B, *C, *hC;
    float *dA, *dB, *dC;
    
    DS_timer timer(5);
    timer.setTimerName(0, (char *)"CUDA Total");
    timer.setTimerName(1, (char *)"Computation on device (GPU)");
    timer.setTimerName(2, (char *)"Memory copy: host -> device");
    timer.setTimerName(3, (char *)"Memory copy: device -> host");
    timer.setTimerName(4, (char *)"Computation on host (CPU)");

    // memory allocation on host
    A = (float *)malloc(sizeof(float) * sizeA);
    B = (float *)malloc(sizeof(float) * sizeB);
    C = (float *)malloc(sizeof(float) * sizeC);
    hC = (float *)malloc(sizeof(float) * sizeC);

    // initialize
    memset(A, 0, sizeof(float) * sizeA);
    memset(B, 0, sizeof(float) * sizeB);
    memset(C, 0, sizeof(float) * sizeC);
    memset(hC, 0, sizeof(float) * sizeC);

    // set data
    for(int i = 0; i < sizeA; i++)
        A[i] = ((rand() % 10) + ((rand() % 100) / 100.0));

    for(int i = 0; i < sizeB; i++)
        B[i] = ((rand() % 10) + ((rand() % 100) / 100.0)); 

    // matrix multiplication on host (CPU)
    timer.onTimer(4);
    
	for (int row = 0; row < m; row++)
		for (int col = 0; col < n; col++)
	        hC[row * n + col] = A[row * n + col] + B[row * n + col];
	
    timer.offTimer(4);

    // memory allocation on device
    cudaMalloc(&dA, sizeof(float) * sizeA);
    cudaMalloc(&dB, sizeof(float) * sizeB);
    cudaMalloc(&dC, sizeof(float) * sizeC);

    // initialize
    cudaMemset(dA, 0, sizeof(float) * sizeA);
    cudaMemset(dB, 0, sizeof(float) * sizeB);
    cudaMemset(dC, 0, sizeof(float) * sizeC);

    timer.onTimer(0);
    
    // memory copy (host -> device)
    timer.onTimer(2);
    cudaMemcpy(dA, A, sizeof(float) * sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(float) * sizeB, cudaMemcpyHostToDevice);
    timer.offTimer(2);

    // kernel call (matrix multiplication on device (GPU))
    timer.onTimer(1);
    // problem 2
    cudaDeviceSynchronize();
    timer.offTimer(1);
    
    // memory copy (device -> host)
    timer.onTimer(3);
    cudaMemcpy(C, dC, sizeof(float) * sizeC, cudaMemcpyDeviceToHost);
    timer.offTimer(3);

    timer.offTimer(0);
    
    timer.printTimer();
    
    // memory deallocation on device
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    // check results
    result = memcmp(C, hC, sizeof(float) * sizeC);

    if (result == 0)
        printf("The matrix addition on the device (GPU) is the same as the the host (CPU)\n");
    else {
        printf("The matrix addition on the device (GPU) is not the same as the the host (CPU)\n");
        
        for(int i = 0; i < sizeC; i++)
            if(C[i] != hC[i])
                printf("C[%d] and hC [%d] is different (error: %.2f)\n", i, i, C[i] - hC[i]);
    }


    // memory deallocation on host
    free(A);
    free(B);
    free(C);
    free(hC);

    return 0;
}
