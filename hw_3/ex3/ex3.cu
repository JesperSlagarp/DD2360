
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>


#define NUM_BINS 4096

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {

//@@ Insert code below to compute histogram of input using shared memory and atomics
  __shared__ unsigned int shared_bins[NUM_BINS];
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
  if (idx < num_elements) {
    atomicAdd(&shared_bins[input[idx]], 1);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
    atomicAdd(&bins[i], shared_bins[i]);
  }
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {

//@@ Insert code below to clean up bins that saturate at 127
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_bins && bins[idx] > 127) {
    bins[idx] = 127;
  }
}

int main(int argc, char **argv) {
  
  int inputLength;
  unsigned int *hostInput;
  unsigned int *hostBins;
  unsigned int *resultRef;
  unsigned int *deviceInput;
  unsigned int *deviceBins;

  //@@ Insert code below to read in inputLength from args

  inputLength = atoi(argv[1]);
  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput = (unsigned int*) malloc(inputLength * sizeof(unsigned int));
  hostBins = (unsigned int*) malloc(NUM_BINS * sizeof(unsigned int));
  resultRef = (unsigned int*) malloc(NUM_BINS * sizeof(unsigned int));
  
  //@@ Insert code below to initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
  srand(time(0));
  for (int i = 0; i < inputLength; i++) {
    hostInput[i] = rand() % NUM_BINS;
  }

  //@@ Insert code below to create reference result in CPU
  for (int i = 0; i < inputLength; i++) {
    resultRef[hostInput[i]] += 1;
  }

  for (int i = 0; i < NUM_BINS; i++) {
    if (resultRef[i] > 127) { resultRef[i] = 127; }
  }

  /*for (int i = 0; i < NUM_BINS; i++) {
    printf("%d, ", resultRef[i]);
  }*/

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput, inputLength * sizeof(unsigned int));
  cudaMalloc(&deviceBins, NUM_BINS * sizeof(unsigned int));

  //@@ Insert code to Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceBins, hostBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyHostToDevice);

  //@@ Insert code to initialize GPU results


  //@@ Initialize the grid and block dimensions here
  dim3 Db1(128);
  dim3 Dg1((inputLength + Db1.x - 1) / Db1.x);

  //@@ Launch the GPU Kernel here
  histogram_kernel<<<Dg1, Db1>>>(deviceInput, deviceBins,
                                 inputLength,
                                 NUM_BINS);
  cudaDeviceSynchronize();

  //@@ Initialize the second grid and block dimensions here
  dim3 Db2(128);
  dim3 Dg2((NUM_BINS + Db2.x - 1) / Db2.x);

  //@@ Launch the second GPU Kernel here
  convert_kernel<<<Dg2, Db2>>>(deviceBins, NUM_BINS);
  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference
  /*printf("\n\n");
  printf("[");
  for (int i = 0; i < NUM_BINS; i++) {
    printf("%d, ", hostBins[i]);
  }
  printf("]\n");
  printf("\n\n");
  printf("[");
  for (int i = 0; i < NUM_BINS; i++) {
    printf("%d, ", resultRef[i]);
  }*/
  //printf("]\n");
  bool equal = true;
  for (int i = 0; i < NUM_BINS; i++) {
    if(resultRef[i] != hostBins[i]) {
      equal = false;
      break;
    }
  }
  equal ? printf("Results equal to resultRef (Success)\n\n") : printf("Results not equal to resultRef (Fail)\n\n");

  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceBins);

  //@@ Free the CPU memory here
  free(hostInput);
  free(hostBins);
  free(resultRef);

  return 0;
}

