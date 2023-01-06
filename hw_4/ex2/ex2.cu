#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DataType double
#define TPB 128

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
  //@@ Insert code to implement vector addition here
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < len) {
    out[id] = in1[id] + in2[id];
  }
}

DataType randRange(DataType min, DataType max) {
  DataType random = ((DataType) rand()) / RAND_MAX;
  DataType range = (max - min) * random;
  return min + range;
}

cudaEvent_t t_start, t_end;
void timer_start(){
  cudaEventCreate(&t_start);
  cudaEventCreate(&t_end);
  cudaEventRecord(t_start, 0);
}
void timer_stop(const char* info){
  float time;
  cudaEventRecord(t_end, 0);
  cudaEventSynchronize(t_end);
  cudaEventElapsedTime(&time, t_start, t_end);
  printf("Timing - %s. \t\tElasped %.0f microseconds \n", info, time * 1000);
}

int main(int argc, char **argv) {
  timer_start();
  
  DataType *hostInput1;
  DataType *hostInput2;
  DataType *hostOutput;
  DataType *resultRef;
  DataType *deviceInput1;
  DataType *deviceInput2;
  DataType *deviceOutput;

  //@@ Insert code below to read in inputLength from args
  const int inputLength = atoi(argv[1]);

  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  hostInput1 = (DataType*) malloc(inputLength * sizeof(DataType));
  hostInput2 = (DataType*) malloc(inputLength * sizeof(DataType));
  hostOutput = (DataType*) malloc(inputLength * sizeof(DataType));
  resultRef = (DataType*) malloc(inputLength * sizeof(DataType));
  
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  srand(time(0));

  for(int i = 0; i < inputLength; i++) {
    DataType v = randRange(-1000, 1000);
    DataType u = randRange(-1000, 1000);
    hostInput1[i] = v;
    hostInput2[i] = u;
    resultRef[i] = v + u;
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceInput1, inputLength*sizeof(DataType));
  cudaMalloc(&deviceInput2, inputLength*sizeof(DataType));
  cudaMalloc(&deviceOutput, inputLength*sizeof(DataType));

  //@@ Insert code to below to Copy memory to the GPU here
  cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);

  //@@ Initialize the 1D grid and block dimensions here

  int Db = TPB;
  int Dg = (inputLength + TPB - 1) / TPB;

  //@@ Launch the GPU Kernel here
  vecAdd<<<Dg + 1, Db>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(DataType), cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference
  bool equal = true;
  for(int i = 0; i < inputLength; i++) {
    if(resultRef[i] != hostOutput[i]) {
      equal = false;
      break;
    }
  }
  equal ? printf("Results equal to resultRef (Success)\n\n") : printf("Results not equal to resultRef (Fail)\n\n");

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  //@@ Free the CPU memory here
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
  free(resultRef);

  //@ Print time
  timer_stop("Time: ");

  return 0;
}