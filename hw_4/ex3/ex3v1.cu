#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DataType double

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  //@@ Insert code to implement matrix multiplication here
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < numARows && j < numBColumns) {
    C[i * numBColumns + j] = 0;
    for(int k = 0; k < numAColumns; k++) {
      C[i * numBColumns + j] += A[i * numAColumns + k] * B[k * numBColumns + j];
    }
  }
}

DataType randRange(DataType min, DataType max) {
  DataType random = ((DataType) rand()) / RAND_MAX;
  DataType range = (max - min) * random;
  return min + range;
}

void printMatrix(DataType * matrix, int rows, int columns) {
  for(int i = 0; i < rows; i++) {
    printf("|");
    for(int j = 0; j < columns; j++) {
      printf("%f|",matrix[i * columns + j]);
    } 
    printf("\n");
  } printf("\n");
}

int main(int argc, char **argv) {
  
  DataType *hostA; // The A matrix
  DataType *hostB; // The B matrix
  DataType *hostC; // The output C matrix
  DataType *resultRef; // The reference result
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;
  int numCColumns;

  //@@ Insert code below to read in numARows, numAColumns, numBColumns from args

  numARows = atoi(argv[1]);
  numAColumns = atoi(argv[2]);
  numBRows = atoi(argv[3]);
  numBColumns = atoi(argv[4]);
  numCRows = numARows;
  numCColumns = numBColumns;
  printf("Input matrix dim (%d x %d) (%d x %d) (%d x %d)\n", numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  
  //@@ Insert code below to allocate Host memory for input and output

  hostA = (DataType*) malloc(numARows * numAColumns * sizeof(DataType));
  hostB = (DataType*) malloc(numBRows * numBColumns * sizeof(DataType));
  hostC = (DataType*) malloc(numCRows * numCColumns * sizeof(DataType));
  resultRef = (DataType*) malloc(numCRows * numCColumns * sizeof(DataType));
  
  //@@ Insert code below to initialize hostA and hostB to random numbers, and create reference result in CPU
  srand(time(0));
  for(int i = 0; i < numARows * numAColumns; i++) {
    hostA[i] = randRange(-1000, 1000);
  }
  for(int i = 0; i < numBRows * numBColumns; i++) {
    hostB[i] = randRange(-1000, 1000);
  }
  for(int i = 0; i < numCRows; i++) {
    for(int j = 0; j < numBColumns; j++) {
      resultRef[i * numBColumns + j] = 0;
      for(int k = 0; k < numAColumns; k++) {
        resultRef[i * numBColumns + j] += hostA[i * numAColumns + k] * hostB[k * numBColumns + j];
      }
    }
  }

  //@@ Insert code below to allocate GPU memory here
  cudaMalloc(&deviceA, numARows * numAColumns * sizeof(DataType));
  cudaMalloc(&deviceB, numBRows * numBColumns * sizeof(DataType));
  cudaMalloc(&deviceC, numCRows * numCColumns * sizeof(DataType));

  //@@ Insert code to below to Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(DataType), cudaMemcpyHostToDevice);
  cudaMemcpy(deviceC, hostC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 Db(32,32);
  dim3 Dg((numCRows + Db.x - 1) / Db.x, (numCColumns + Db.y - 1) / Db.y);
  
  //@@ Launch the GPU Kernel here
  gemm<<<Dg, Db>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(DataType), cudaMemcpyDeviceToHost);

  //@@ Insert code below to compare the output with the reference
  bool equal = true;
  for(int i = 0; i < numCRows; i++) {
    for(int j = 0; j < numCColumns; j++) {
      if(abs(resultRef[i * numCColumns + j] - hostC[i * numCColumns + j]) > 1) {
        printf("%f, %f\n", resultRef[i * numCColumns + j],hostC[i * numCColumns + j]);
        equal = false;
        goto loopend;
      }
    }
  }
  loopend:
  equal ? printf("Results equal to resultRef (Success)\n\n") : printf("Results not equal to resultRef (Fail)\n\n");

  //printMatrix(hostC, numCRows, numCColumns);
  //printMatrix(resultRef, numCRows, numCColumns);

  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  //@@ Free the CPU memory here
  free(hostA);
  free(hostB);
  free(hostC);
  free(resultRef);

  return 0;
}