// Create other necessary functions here
__global__ void ReducedMat_MUL(int *d_A, int *d_B, int *d_C, int MatSize){
    /*
        Per thread one entry of C matrix is calculated 
        -> Matrix is divided into different threadBlocks
        -> Each thread in threadBlock will handle each entry of C matrix
        -> No data sharing is needed thus dividing matrix in threadBlocks have no effect performance  
    */

    // Row and Column entries of C matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int C_val = 0;
    int C_size = MatSize / 2;

    // Calculating row and column for reduced matrice multiplication
    int row1 = 2 * row;
    int row2 = 2 * row + 1;
    int col1 = 2 * col;
    int col2 = 2 * col + 1;
    int sum1=0, sum2=0, sum3=0, sum4=0;
    for(int k = 0; k < MatSize; k++){
        // (row1,col1), (row1,col2), (row2,col2), (row2,col1)
        
        // Sum1 ----> Calculating dot product of (row1,col1)
        int posA1 = d_A[row1 * MatSize + k];
        int posB1 = d_B[k * MatSize + col1];
        sum1 += posA1 * posB1;
        
        // Sum2 ----> Calculating dot product of (row1,col2)
        int posA2 = d_A[row1 * MatSize + k];
        int posB2 = d_B[k * MatSize + col2];
        sum2 += posA2 * posB2;
        
        // Sum3 ----> Calculating dot product of (row2,col1)
        int posA3 = d_A[row2 * MatSize + k];
        int posB3 = d_B[k * MatSize + col1];
        sum3 += posA3 * posB3;
        
        // Sum4 ----> Calculating dot product of (row2,col2)
        int posA4= d_A[row2 * MatSize + k];
        int posB4 = d_B[k * MatSize + col2];
        sum4 += posA4 * posB4;
    }

    // Reduced Matrix multiplication one entry
    C_val = sum1 + sum2 + sum3 + sum4;
    d_C[row * C_size + col] = C_val;

    // printf("%d %d\n",row * C_size + col,C_val);
}

// Fill in this function
void gpuThread(int N, int *matA, int *matB, int *output)
{
    //Calculate size of matrix in bytes
    size_t size = N * N * sizeof(int);
    
    //Pointers for device matrices
    int *d_A, *d_B, *d_C;

    //Allocate device memory for matrice A
    cudaMalloc((void**)&d_A, size);
    cudaMemcpy(d_A, matA, size, cudaMemcpyHostToDevice);

    //Allocate device memory for matrice B
    cudaMalloc((void**)&d_B, size);
    cudaMemcpy(d_B, matB, size, cudaMemcpyHostToDevice);

    //Allocate device memory for d_C
    cudaMalloc((void**)&d_C, size/4);  // size/4

    //Launch kernel on device
    int blockSize = 16; //1,2,4,8,16,32, ----> 64,128,256,512(if possible)
    // int reducedN = N/2;

    cout << "\nBlock size: " << blockSize << endl; 
    dim3 gridDim(N/(2*blockSize), N/(2*blockSize));
    dim3 blockDim(blockSize, blockSize);

    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
    ReducedMat_MUL<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
	cudaDeviceSynchronize();


    //Copy result from device to host
    cudaMemcpy(output, d_C, size/4, cudaMemcpyDeviceToHost); // size/4

    float time= 0;
	cudaEventElapsedTime(&time, start, stop);
	
	printf("GPU Elapsed Time :  %f\n",time);

    //Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    //Free host memory
    cudaFree(matA);
    cudaFree(matB); 

}
