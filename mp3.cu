/*
 *                         Tiled Matrix Multiplication
 *             (MP2, Fall 2014, GPU Programming/Auburn University)
 *
 *   Compile with -DTILE_WIDTH=16 (for example) to change the tile size.
 *   Compile with -DSEED=12 (for example) to seed the random number generator.
 */

 #include <assert.h>
 #include <cuda.h>
 #include <stdio.h>
 #include <math.h>
 #include <iostream>
 #include <fstream>
 /* Usage message displayed when invalid command line arguments are supplied */
 #define USAGE \
     "MP2 generates a random (m x k) matrix M and (k x n) matrix N\n" \
     "and multiplies M by N using tiled matrix multiplication.\n" \
     "The values of m, k, and n must be >= 1.\n" \
     "\n" \
     "Usage: mp2 m k n\n"
 
 /* Tile size -- define here if not defined using the -D compiler flag */
 #ifndef TILE_WIDTH
 #  define TILE_WIDTH 16
 #endif
 
 /* Seed for the random number generator -- define here if not using -D */
 #ifndef SEED
 #  define SEED 1
 #endif
 
 /* Maximum difference allowed between the GPU and CPU result matrices */
 #define EPSILON 1e-2
 
 /* If a CUDA call fails, display an error message and exit */
 #define CUDA_CHECK(e) { \
     cudaError_t err = (e); \
     if (err != cudaSuccess) \
     { \
         fprintf(stderr, "CUDA error: %s, line %d, %s: %s\n", \
             __FILE__, __LINE__, #e, cudaGetErrorString(err)); \
         exit(EXIT_FAILURE); \
     } \
 }
 
 /* assert() is only supported on devices of compute capability >= 2.0 */
 #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
 #  undef  assert
 #  define assert(arg)
 #endif

void getDistance(float *h_M, float *h_P, int m, int k, int col, int row){
    float expected = 0.0;
    for (int i = 0; i < k; i++)
    {
        expected += pow(h_M[row*k+i] - h_M[col*k+i], 2);
    }
    expected = sqrt(expected);
    h_P[row*m+col] = expected;

 }
void argMin(float* h_M, int* result, int m, int k, int row){
    float minimum = 1e5;
    int pos = -1;
    for(int col = 0; col < m; col++){
        if (h_M[row*m+col] < minimum){
            pos = col;
            minimum = h_M[row*m+col];
        }
    }
    result[row] = pos;
}
 __global__ static void argMin_gpu(float* h_M, int* result, int m, int k){
    assert(blockDim.x == TILE_WIDTH && blockDim.y == TILE_WIDTH);

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    if(row >= m)
        return;
    float minimum = 1e5;
    int pos = -1;
    for(int col = 0; col < m; col++){
        if (h_M[row*m+col] < minimum){
            pos = col;
            minimum = h_M[row*m+col];
        }
    }
    result[row] = pos;
 }
 __global__ static void getDistance_gpu(float *d_M, float *d_P, int m, int k) {
    assert(blockDim.x == TILE_WIDTH && blockDim.y == TILE_WIDTH);

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col= blockIdx.x  * TILE_WIDTH + threadIdx.x;
    if(row >= m || col >= m)
        return;
    if(row == col){
        d_P[row*m+col] = 100;
        return;
    }

    float expected = 0.0;
    for (int i = 0; i < k; i++)
    {
        expected += pow(d_M[row*k+i] - d_M[col*k+i], 2);
    }
    expected = sqrt(expected);
    d_P[row*m+col] = expected;
}

 /* Displays one row of the given matrix */
 static void printRow(int row, float *matrix, int cols)
 {
     printf("[");
     if (cols >= 1) printf(" %3.3f", matrix[row*cols+0]);
     if (cols >= 2) printf(" %3.3f", matrix[row*cols+1]);
     if (cols >= 3) printf(" %3.3f", matrix[row*cols+2]);
     if (cols >= 6) printf(" ...");
     if (cols >= 5) printf(" %3.3f", matrix[row*cols+(cols-2)]);
     if (cols >= 4) printf(" %3.3f", matrix[row*cols+(cols-1)]);
     printf(" ]\n");
 }
 
 /* Displays the given matrix */
 static void printMatrix(float *matrix, int rows, int cols)
 {
     if (rows >= 1) printRow(0, matrix, cols);
     if (rows >= 2) printRow(1, matrix, cols);
     if (rows >= 3) printRow(2, matrix, cols);
     if (rows >= 6) printf("  ...\n");
     if (rows >= 5) printRow(rows-2, matrix, cols);
     if (rows >= 4) printRow(rows-1, matrix, cols);
 }
 
 /* Program entrypoint.  Invoke with three command line arguments: m k n */
 int main()
 {

    printf("%d, %d, %d, %d\n", sizeof(long), sizeof(long long), sizeof(bool), sizeof(char));
     /* Get command line arguments; save as m, k, and n */
     int m = 100;
     int k = 128;

     if (m < 1 || k < 1)
     {
         fprintf(stderr, USAGE);
         fprintf(stderr, "Invalid value for m or k (%d, %d)\n",
             m, k);
         return EXIT_FAILURE;
     }
     printf("using (%d x %d) tiles.\n", TILE_WIDTH, TILE_WIDTH);
 
     /********************************************/
     /* M is (m x k), P is (m x m) */
     /********************************************/
 
     /* Compute number of bytes needed to stores matrices M and P */
     size_t bytesForM = m * k * sizeof(float);
     size_t bytesForP = m * m * sizeof(float);
 
     /* Allocate host memory for matrices */
     float *h_M, *h_P;
     float *result = new float[m*m];
     int *index = new int[m];

     h_M = (float *)malloc(bytesForM);
     h_P = (float *)malloc(bytesForP);

     if (h_M == NULL || h_P == NULL)
     {
         fprintf(stderr, "Unable to allocate host memory\n");
         return EXIT_FAILURE;
     }
 
     /* Allocate device memory for matrices */
     float *d_M, *d_P;
     int *d_index;
     CUDA_CHECK(cudaMalloc((void **)&d_M, bytesForM));
     CUDA_CHECK(cudaMalloc((void **)&d_P, bytesForP));
     CUDA_CHECK(cudaMalloc((void **)&d_index, m*sizeof(int)));
 
     /* Fill M (on host) */
 
     


     std::cout << "Loading matrices...\n";
     std::ifstream in1, in2;
	in1.open("descriptor.txt");
	for (int i = 0; i < m*k; ++i)
		in1 >> h_M[i];
    in1.close();

    printf("M =\n"); printMatrix(h_M, m, k);

     /* Copy M to device global memory */
     CUDA_CHECK(cudaMemcpy(d_M, h_M, bytesForM, cudaMemcpyHostToDevice));
 
     /* Launch the CUDA kernel */
     dim3 dimGrid((m+TILE_WIDTH-1)/TILE_WIDTH, (m+TILE_WIDTH-1)/TILE_WIDTH);
     dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
 
    printf("matMul called from host\n");
     
     getDistance_gpu<<<dimGrid, dimBlock>>>(d_M, d_P, m, k);
     argMin_gpu<<<dimGrid, dimBlock>>>(d_P, d_index, m, k);
     CUDA_CHECK(cudaDeviceSynchronize());
 
     /* Copy result matrix from device global memory back to host memory */
     CUDA_CHECK(cudaMemcpy(h_P, d_P, bytesForP, cudaMemcpyDeviceToHost));
     CUDA_CHECK(cudaMemcpy(index, d_index, m*sizeof(int), cudaMemcpyDeviceToHost));
           
    printf(" product received from host\n");

    printf("P =\n"); printMatrix(h_P, m, m);

     std::ofstream out, out2;
     out2.open("matrix.txt");
     for (int i = 0; i < 100; i++){
        for (int j = 0; j < m; j++){
            if (h_P[i*m+j] < 0.3)
                out2 << j+1 << " ";
        }
        out2 << std::endl;
        
     }
     out2.close();

    /*
     
     for (int row = 0; row < m; row++)
     {
         for (int col = 0; col < m; col++)
         {
            getDistance(h_M, result, m, k, col, row);
         }
    }
     
     printf("\nExpected matrix:\n");
     printMatrix(result, m, m);

     printf("\n");
     for (int i = 0; i < m; i++){
        printf("%d ", index[i]);
    }
    printf("\n");
    */
     /* Free device global memory */
     CUDA_CHECK(cudaFree(d_M));
     CUDA_CHECK(cudaFree(d_P));
     CUDA_CHECK(cudaFree(d_index));

 
     /* Free host memory */
     free(h_M);
     free(h_P);
     free(index);
     free(result);
 
     /* Reset the device (unnecessary if not profiling, but good practice) */
     CUDA_CHECK(cudaDeviceReset());
 
     printf("Done\n");
     system("Pause");
     return EXIT_SUCCESS;
 }
 