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
  const int S_seg = atoi(argv[2]);

  printf("The input length is %d\n", inputLength);
  
  //@@ Insert code below to allocate Host memory for input and output
  cudaMallocHost((void **)&hostInput1, inputLength * sizeof(DataType));
  cudaMallocHost((void **)&hostInput2, inputLength * sizeof(DataType));
  cudaMallocHost((void **)&hostOutput, inputLength * sizeof(DataType));
  resultRef = (DataType*) malloc(inputLength * sizeof(DataType));

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc((void **)&deviceInput1, inputLength * sizeof(DataType));
  cudaMalloc((void **)&deviceInput2, inputLength * sizeof(DataType));
  cudaMalloc((void **)&deviceOutput, inputLength * sizeof(DataType));
  
  //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
  srand(time(0));
  for(int i = 0; i < inputLength; i++) {
    DataType v = randRange(-1000, 1000);
    DataType u = randRange(-1000, 1000);
    hostInput1[i] = v;
    hostInput2[i] = u;
    resultRef[i] = v + u;
  }

  // Declare 4 streams for asynchronous copying
  cudaStream_t streams[4];
  for (int i = 0; i < 4; i++) {
    cudaStreamCreate(&streams[i]);
  }

  //@@ Initialize the 1D grid and block dimensions here

  int Db = TPB;
  int Dg = (S_seg + TPB - 1) / TPB;

  //@@ Launch the GPU Kernel here
  int runs = inputLength / S_seg + 1;
  for(int i = 0; i < runs; i++) {
    int stream = i % 4;
    int offset = i * S_seg;
    int length = min(S_seg, inputLength - offset);
    int bytes = min(S_seg, inputLength - offset) * sizeof(DataType);
    cudaMemcpyAsync(deviceInput1 + offset, hostInput1 + offset, bytes, cudaMemcpyHostToDevice, streams[stream]);
    cudaMemcpyAsync(deviceInput2 + offset, hostInput2 + offset, bytes, cudaMemcpyHostToDevice, streams[stream]);
    vecAdd<<<Dg, Db, 0, streams[stream]>>>(deviceInput1 + offset, deviceInput2 + offset, deviceOutput + offset, length);
    cudaMemcpyAsync(hostOutput + offset, deviceOutput + offset, bytes, cudaMemcpyDeviceToHost, streams[stream]);
  }
  for (int i = 0; i < 4; i++) {
      cudaStreamSynchronize(streams[i]);
  }
  
  //@@ Insert code below to compare the output with the reference
  bool equal = true;
  for(int i = 0; i < inputLength; i++) {
    if(resultRef[i] != hostOutput[i]) {
      equal = false;
      break;
    }
  }
  equal ? printf("Results equal to resultRef (Success)\n\n") : printf("Results not equal to resultRef (Fail)\n\n");

  /*for(int i = 0; i < inputLength; i++) {
    printf("%f : %f\n", resultRef[i], hostOutput[i]);
  }*/

  for (int i = 0; i < 4; i++) {
    cudaStreamDestroy(streams[i]);
  }

  //@@ Free the GPU memory here
  cudaFree(deviceInput1);
  cudaFree(deviceInput2);
  cudaFree(deviceOutput);

  //@@ Free the CPU memory here
  cudaFree(hostInput1);
  cudaFree(hostInput2);
  cudaFree(hostOutput);
  free(resultRef);

   //@@ Print time
  timer_stop("Time: ");

  return 0;
}