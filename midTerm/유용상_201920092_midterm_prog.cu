#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DS_timer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_DATA 8192

// problem 1
__global__ void vecMulDiv(double *a, double *b, double *c, double *d, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        d[idx] = a[idx] * b[idx] / c[idx];
    }
}
int main(void)
{
    double *a, *b, *c, *d, *hd;
    double *da, *db, *dc, *dd; 
    int memsize = sizeof(double) * NUM_DATA;
    int result;

    DS_timer timer(5);
    timer.setTimerName(0, (char *)"CUDA Total");
    timer.setTimerName(1, (char *)"Computation on device (GPU)");
    timer.setTimerName(2, (char *)"Memory copy: host -> device");
    timer.setTimerName(3, (char *)"Memory copy: device -> host");
    timer.setTimerName(4, (char *)"Computation on host (CPU)");

    // memory allocation on host
    a = (double *)malloc(memsize);
    b = (double *)malloc(memsize);
    c = (double *)malloc(memsize);
    d = (double *)malloc(memsize);
    hd = (double *)malloc(memsize);

    // initialize
    memset(a, 0, memsize);
    memset(b, 0, memsize);
    memset(c, 0, memsize);
    memset(d, 0, memsize);
    memset(hd, 0, memsize);

    // set data
    for(int i = 0; i < NUM_DATA; i++){
        a[i] = rand() % 10;
        b[i] = rand() % 10;
        c[i] = rand() % 10;
    }

    // vector sum on host
    timer.onTimer(4);
    
    for(int i = 0; i < NUM_DATA; i++)
        hd[i] = a[i] * b[i] / c[i];
    timer.offTimer(4);

    timer.onTimer(0);
    cudaMalloc((void **)&da, memsize);
    cudaMalloc((void **)&db, memsize);
    cudaMalloc((void **)&dc, memsize);
    cudaMalloc((void **)&dd, memsize);

    timer.onTimer(2);
    cudaMemcpy(da, a, memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(dc, c, memsize, cudaMemcpyHostToDevice);
    timer.offTimer(2);

    timer.onTimer(1);
    int blockSize = 256;
    int gridSize = (NUM_DATA + blockSize - 1) / blockSize;
    vecMulDiv<<<gridSize, blockSize>>>(da, db, dc, dd, NUM_DATA);
    cudaDeviceSynchronize();
    timer.offTimer(1);

    timer.onTimer(3);
    cudaMemcpy(d, dd, memsize, cudaMemcpyDeviceToHost);
    timer.offTimer(3);


    // problem 2

    // check results
    result = memcmp(d, hd, memsize);

    if (result ==0)
        printf("The data sum on the device (GPU) is the same as the data sum on the host (CPU)\n");
    else
        printf("The data sum on the device (GPU) is not the same as the data sum on the host (CPU)\n");

    // memory deallocation on host
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    cudaFree(dd);

    free(a);
    free(b);
    free(c);
    free(d);
    free(hd);
    timer.offTimer(0);

    timer.printTimer();
    return 0;
}
