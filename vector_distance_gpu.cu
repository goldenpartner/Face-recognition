/*
 *                         Face Factor Distance
 *             (MP3, Fall 2019, GPU Programming/Yifan Liu)
*/

#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <string>
#include <sstream>
/* Usage message displayed when invalid command line arguments are supplied */
#define USAGE \
    "MP3 takes a (m x k) matrix M \n" \
    "and compute the distance betwen rows and save teh result if two rows' distance is smaller than 0.3\n" \
    "The values of m, k must be >= 1.\n" \
    "\n" \
    "Usage: mp3 m k\n"
 
/* Tile size*/
#ifndef TILE_WIDTH
#  define TILE_WIDTH 16
#endif

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

 /*getDistance calculate the distance among rows*/
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
 
/* Program entrypoint. */
int main()
{

    /* read in m and k here */
    std::cout << "Loading matrices...\n";
    clock_t begin = clock();
    int m, k;
    std::ifstream in1;
    in1.open("descriptor.txt");
    
    if(in1.is_open())
        printf("File opened successfully\n");
    else
        printf("File opened unsuccessfully\n");
    
    std::string line, temp;
    // read in m and k
    while ((std::getline(in1, line))){
        if (line == "end header")
            break;
        std::istringstream ss(line);
        std::cout << line << std::endl;
        if(line.find("line_number")!=-1)    
            ss >> temp >> m;
        else if(line.find("vector_dimension")!=-1)
            ss >> temp >> k;
    }
    printf("The matrix is %d x %d\n", m, k);
    if (m < 1 || k < 1)
    {
        fprintf(stderr, USAGE);
        fprintf(stderr, "Invalid value for m or k (%d, %d)\n", m, k);
        system("Pause");
        return EXIT_FAILURE;
    }

    size_t bytesForM = m * k * sizeof(float);
    float *h_M = (float *)malloc(bytesForM);
    /* Fill M (on host) */
    for (int i = 0; i < m*k; ++i)
        in1 >> h_M[i];
    in1.close();
    
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    printf("Reading input file took %f seconds\n", elapsed_secs);
    printf("M =\n"); printMatrix(h_M, m, k);



    printf("using (%d x %d) tiles.\n", TILE_WIDTH, TILE_WIDTH);
    /********************************************/
    /* M is (m x k), P is (m x m) */
    /********************************************/
    size_t bytesForP = m * m * sizeof(float);
    float *h_P = (float *)malloc(bytesForP);

    if (h_M == NULL || h_P == NULL)
    {
        fprintf(stderr, "Unable to allocate host memory\n");
        system("Pause");
        return EXIT_FAILURE;
    }
    
    /* Allocate device memory for matrices */
    float *d_M, *d_P;
    CUDA_CHECK(cudaMalloc((void **)&d_M, bytesForM));
    CUDA_CHECK(cudaMalloc((void **)&d_P, bytesForP));

    /* Copy M to device global memory */
    CUDA_CHECK(cudaMemcpy(d_M, h_M, bytesForM, cudaMemcpyHostToDevice));
 
    /* Launch the CUDA kernel */
    dim3 dimGrid((m+TILE_WIDTH-1)/TILE_WIDTH, (m+TILE_WIDTH-1)/TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
 
    printf("matMul called from host\n");
     
    getDistance_gpu<<<dimGrid, dimBlock>>>(d_M, d_P, m, k);
    CUDA_CHECK(cudaDeviceSynchronize());
 
    /* Copy result matrix from device global memory back to host memory */
    CUDA_CHECK(cudaMemcpy(h_P, d_P, bytesForP, cudaMemcpyDeviceToHost));
           
    printf(" product received from host\n");
    printf("P =\n"); printMatrix(h_P, m, m);

    printf("Saving result\n");
     std::ofstream out;
     out.open("matrix.txt");
     for (int i = 0; i < m; i++){
        for (int j = 0; j < m; j++){
            if (h_P[i*m+j] < 0.3)
                out << j+1 << " ";
        }
        out << std::endl;
    }
    out.close();


    /* Free device global memory */
    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_P));

    /* Free host memory */
    free(h_M);
    free(h_P);

    /* Reset the device (unnecessary if not profiling, but good practice) */
    CUDA_CHECK(cudaDeviceReset());
 
    printf("Done\n");
    system("Pause");
    return EXIT_SUCCESS;
 }
 