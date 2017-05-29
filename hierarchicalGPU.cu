#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <fstream>
#include <cfloat>
#include <string>
#include <sstream>
#include <assert.h>
#define CAMPAIGN_DISTANCE_MANHATTAN /** < Type of distance metric */
#define THREADSPERBLOCK 300         /** < Threads per block (tpb) */
#define FLOAT_TYPE float            /** < Precision of floating point numbers */
using namespace std;
namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}
template <typename T>
T allocDeviceMemory(int memSize, T data)
{
    T retVal;
    cudaMalloc((void**) &retVal, memSize);
    cudaMemcpy(retVal, data, memSize, cudaMemcpyHostToDevice); // copy data from host to device
    return retVal;
}

template <typename T>
T allocDeviceMemory(int memSize)
{
    T retVal;
    cudaMalloc((void**) &retVal, memSize);
    return retVal;
}

template <class T>
__device__ static T distanceComponentGPU(T *elementA, T *elementB)
{
T dist = 0.0f;
#ifdef CAMPAIGN_DISTANCE_EUCLIDEAN_SQUARED
    dist = elementA[0] - elementB[0];
    dist = dist * dist;
#elif defined(CAMPAIGN_DISTANCE_EUCLIDEAN)
    dist = elementA[0] - elementB[0];
    dist = dist * dist;
#elif defined(CAMPAIGN_DISTANCE_MANHATTAN)
    dist = fabs(elementA[0] - elementB[0]);
#elif defined(CAMPAIGN_DISTANCE_CHEBYSHEV)
    dist = fabs(elementA[0] - elementB[0]);
#else
#error "No distance defined, add #define x to program's header with x = CAMPAIGN_DISTANCE_EUCLIDEAN_SQUARED, CAMPAIGN_DISTANCE_EUCLIDEAN, CAMPAIGN_DISTANCE_MANHATTAN, or CAMPAIGN_DISTANCE_CHEBYSHEV");
#endif
return dist;
}

template <class T>
__device__ static T distanceFinalizeGPU(int D, T *components)
{
    T dist = 0.0f;
#ifdef CAMPAIGN_DISTANCE_EUCLIDEAN_SQUARED
    for (unsigned int cnt = 0; cnt < D; cnt++) dist += components[cnt];
#elif defined(CAMPAIGN_DISTANCE_EUCLIDEAN)
    for (unsigned int cnt = 0; cnt < D; cnt++) dist += components[cnt];
    dist = sqrt(dist);
#elif defined(CAMPAIGN_DISTANCE_MANHATTAN)
    for (unsigned int cnt = 0; cnt < D; cnt++) dist += components[cnt];
#elif defined(CAMPAIGN_DISTANCE_CHEBYSHEV)
    for (unsigned int cnt = 0; cnt < D; cnt++) dist = max(dist, components[cnt]);
#else
#error "No distance defined, add #define x to program's header with x = CAMPAIGN_DISTANCE_EUCLIDEAN_SQUARED, CAMPAIGN_DISTANCE_EUCLIDEAN, CAMPAIGN_DISTANCE_MANHATTAN, or CAMPAIGN_DISTANCE_CHEBYSHEV");
#endif
    return dist;
}

template <unsigned int BLOCKSIZE, class T, class U>
__device__ static void reduceMinTwo(int tid, T *s_A, U *s_B)
{
    if (BLOCKSIZE >= 1024) { if (tid < 512)
        {
            if (s_A[tid + 512] == s_A[tid]) s_B[tid] = min(s_B[tid], s_B[tid + 512]);
            if (s_A[tid + 512] <  s_A[tid]) { s_A[tid] = s_A[tid + 512]; s_B[tid] = s_B[tid + 512]; }
        } __syncthreads(); }
    if (BLOCKSIZE >=  512) { if (tid < 256)
        {
            if (s_A[tid + 256] == s_A[tid]) s_B[tid] = min(s_B[tid], s_B[tid + 256]);
            if (s_A[tid + 256] <  s_A[tid]) { s_A[tid] = s_A[tid + 256]; s_B[tid] = s_B[tid + 256]; }
        } __syncthreads(); }
    if (BLOCKSIZE >=  256) { if (tid < 128)
        {
            if (s_A[tid + 128] == s_A[tid]) s_B[tid] = min(s_B[tid], s_B[tid + 128]);
            if (s_A[tid + 128] <  s_A[tid]) { s_A[tid] = s_A[tid + 128]; s_B[tid] = s_B[tid + 128]; }
        } __syncthreads(); }
    if (BLOCKSIZE >=  128) { if (tid <  64)
        {
            if (s_A[tid +  64] == s_A[tid]) s_B[tid] = min(s_B[tid], s_B[tid +  64]);
            if (s_A[tid +  64] <  s_A[tid]) { s_A[tid] = s_A[tid +  64]; s_B[tid] = s_B[tid +  64]; }
        } __syncthreads(); }

    if (tid < 32)
    {
        if (BLOCKSIZE >= 64) { if (s_A[tid + 32] == s_A[tid]) s_B[tid] = min(s_B[tid], s_B[tid + 32]); if (s_A[tid + 32] < s_A[tid]) { s_A[tid] = s_A[tid + 32]; s_B[tid] = s_B[tid + 32]; } }
        if (BLOCKSIZE >= 32) { if (s_A[tid + 16] == s_A[tid]) s_B[tid] = min(s_B[tid], s_B[tid + 16]); if (s_A[tid + 16] < s_A[tid]) { s_A[tid] = s_A[tid + 16]; s_B[tid] = s_B[tid + 16]; } }
        if (BLOCKSIZE >= 16) { if (s_A[tid +  8] == s_A[tid]) s_B[tid] = min(s_B[tid], s_B[tid +  8]); if (s_A[tid +  8] < s_A[tid]) { s_A[tid] = s_A[tid +  8]; s_B[tid] = s_B[tid +  8]; } }
        if (BLOCKSIZE >=  8) { if (s_A[tid +  4] == s_A[tid]) s_B[tid] = min(s_B[tid], s_B[tid +  4]); if (s_A[tid +  4] < s_A[tid]) { s_A[tid] = s_A[tid +  4]; s_B[tid] = s_B[tid +  4]; } }
        if (BLOCKSIZE >=  4) { if (s_A[tid +  2] == s_A[tid]) s_B[tid] = min(s_B[tid], s_B[tid +  2]); if (s_A[tid +  2] < s_A[tid]) { s_A[tid] = s_A[tid +  2]; s_B[tid] = s_B[tid +  2]; } }
        if (BLOCKSIZE >=  2) { if (s_A[tid +  1] == s_A[tid]) s_B[tid] = min(s_B[tid], s_B[tid +  1]); if (s_A[tid +  1] < s_A[tid]) { s_A[tid] = s_A[tid +  1]; s_B[tid] = s_B[tid +  1]; } }
    }
}


__global__ static void calcDistanceMatrix_CUDA(int N, int D, FLOAT_TYPE *X, int *NEIGHBOR, FLOAT_TYPE *NEIGHBORDIST)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;

    if (t < ceilf((FLOAT_TYPE) N / 2.0))
    {
        int row = t, col = -1, row2 = N - t - 1;
        NEIGHBOR[row] = NEIGHBOR[row2] = -1;
        NEIGHBORDIST[row] = NEIGHBORDIST[row2] = FLT_MAX;
        for (int j = 0; j < N - 1; j++)
        {
            col++;
            if (t == j) { row = row2; col = 0; }
            FLOAT_TYPE distance = 0.0;
            for (int d = 0; d < D; d++) distance += distanceComponentGPU(X + d * N + row, X + d * N + col);
            distance = distanceFinalizeGPU(1, &distance);
            if (distance < NEIGHBORDIST[row])
            {
                NEIGHBOR[row] = col;
                NEIGHBORDIST[row] = distance;
            }
        }
    }
}


__global__ void min_CUDA(unsigned int N, unsigned int iter, FLOAT_TYPE *INPUT, int *INKEY, FLOAT_TYPE *OUTPUT, int *OUTKEY)
{
    extern __shared__ FLOAT_TYPE array[];
    FLOAT_TYPE* s_value = (FLOAT_TYPE*) array;
    int*   s_key   = (int*)   &s_value[blockDim.x];

    unsigned int tid = threadIdx.x;
    unsigned int t   = blockIdx.x*blockDim.x + threadIdx.x;
    s_value[tid] = FLT_MAX;
    s_key  [tid] = 0;

    if (t < N)
    {
        s_value[tid] = INPUT[t];
        s_key  [tid] = (iter == 0) ? t : INKEY[t];
    }
    __syncthreads();

    reduceMinTwo<THREADSPERBLOCK>(tid, s_value, s_key);

    if (tid == 0)
    {
        OUTPUT[blockIdx.x] = s_value[tid];
        OUTKEY[blockIdx.x] = s_key  [tid];
    }
}



int getMinDistIndex(int numClust, FLOAT_TYPE *DISTS, int *INDICES, FLOAT_TYPE *REDUCED_DISTS, int *REDUCED_INDICES)
{
    dim3 block(THREADSPERBLOCK);
    int numBlocks = (int) ceil((FLOAT_TYPE) numClust / (FLOAT_TYPE) THREADSPERBLOCK);
    dim3 gridN(numBlocks);
    int sMem = (sizeof(FLOAT_TYPE) + sizeof(int)) * THREADSPERBLOCK;

    min_CUDA<<<gridN, block, sMem>>>(numClust, 0, DISTS, INDICES, REDUCED_DISTS, REDUCED_INDICES);

    while (numBlocks > 1)
    {
        int nElements = numBlocks;
        numBlocks = (int) ceil((FLOAT_TYPE) nElements / (FLOAT_TYPE) THREADSPERBLOCK);
        dim3 nBlocks(numBlocks);
        min_CUDA<<<nBlocks, block, sMem>>>(nElements, 1, REDUCED_DISTS, REDUCED_INDICES, REDUCED_DISTS, REDUCED_INDICES);
    }

    int *ind = (int*) malloc(sizeof(int));
    cudaMemcpy(ind, REDUCED_INDICES, sizeof(int) * 1, cudaMemcpyDeviceToHost);
    int index = ind[0];

    free(ind);
    return index;
}


__global__ static void mergeElementsInsertAtA_CUDA(int N, int D, int indexA, int indexB, FLOAT_TYPE *X)
{
    int d = blockDim.x * blockIdx.x + threadIdx.x;
    if (d < D)
    {
        X[d * N + indexA] = (X[d * N + indexA] + X[d * N + indexB]) / 2.0;
    }
}



__global__ static void computeAllDistancesToA_CUDA(int N, int numClust, int D, int indexA, FLOAT_TYPE *X, FLOAT_TYPE *DISTS, int *INDICES, FLOAT_TYPE *REDUCED_DISTS, int *REDUCED_INDICES) // includes distance from old B to A, which will later be discarded
{
    int numThreads = blockDim.x;
    extern __shared__ FLOAT_TYPE array[];
    FLOAT_TYPE* s_dist  = (FLOAT_TYPE*) array;
    int*   s_index = (int*  ) &s_dist[numThreads];
    FLOAT_TYPE* s_posA  = (FLOAT_TYPE*) &s_index[numThreads];
    int t = numThreads * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    if (t < numClust) s_dist[tid] = 0.0;

    unsigned int offsetD = 0;
    while (offsetD < D)
    {
        if (offsetD + tid < D) s_posA[tid] = X[(offsetD + tid) * N + indexA];
        __syncthreads();
        for (unsigned int d = offsetD; d < min(offsetD + numThreads, D); d++)
        {
            if (t < numClust) s_dist[tid] += distanceComponentGPU(X + d * N + t, s_posA + d - offsetD);
        }
        offsetD += numThreads;
        __syncthreads();
    }
    if (t < numClust) s_dist[tid] = distanceFinalizeGPU(1, s_dist + tid);
    s_index[tid] = t;
    __syncthreads();

    //index<A covered in parallel red. others updated directly
    if (t > indexA && t < numClust)
    {
        FLOAT_TYPE dist = DISTS[t];
        if (s_dist[tid] == dist) INDICES[t] = min(indexA, INDICES[t]);
        else if (s_dist[tid] < dist)
        {
            DISTS  [t] = s_dist[tid];
            INDICES[t] = indexA;
        }
    }
    if (t >= indexA) s_dist[tid] = FLT_MAX;
    __syncthreads();
    //Th. - Synchronized for index>indexA for index<A using parallel reduction

    reduceMinTwo<THREADSPERBLOCK>(tid, s_dist, s_index);
    if (tid == 0)
    {
        REDUCED_DISTS  [blockIdx.x] = s_dist[tid];
        REDUCED_INDICES[blockIdx.x] = s_index[tid];
    }
}


__global__ static void updateElementA_CUDA(int indexA, FLOAT_TYPE *DISTS, int *INDICES, FLOAT_TYPE *REDUCED_DISTS, int *REDUCED_INDICES)
{
    DISTS  [indexA] = REDUCED_DISTS  [0];
    INDICES[indexA] = REDUCED_INDICES[0];
}



void updateDistanceAndIndexForCluster(int indexA, int sMem, FLOAT_TYPE *DISTS, int *INDICES, FLOAT_TYPE *REDUCED_DISTS, int *REDUCED_INDICES)
{
    dim3 block(THREADSPERBLOCK);
    int numBlocks = (int) ceil((FLOAT_TYPE) (indexA - 1) / (FLOAT_TYPE) THREADSPERBLOCK);
    while (numBlocks > 1)
    {
        dim3 nBlocks((int) ceil((FLOAT_TYPE) numBlocks / (FLOAT_TYPE) THREADSPERBLOCK));
        min_CUDA<<<nBlocks, block, sMem>>>(numBlocks, 1, REDUCED_DISTS, REDUCED_INDICES, REDUCED_DISTS, REDUCED_INDICES);
        numBlocks = nBlocks.x;
    }
    dim3 numB(1);
    updateElementA_CUDA<<<numB, numB>>>(indexA, DISTS, INDICES, REDUCED_DISTS, REDUCED_INDICES);
}



__global__ static void moveCluster_CUDA(int N, int D, int indexB, int indexN, FLOAT_TYPE *X, FLOAT_TYPE *DISTS, int *INDICES)
{
    int d = blockDim.x * blockIdx.x + threadIdx.x;
    if (d < D)
    {
        X[d * N + indexB] = X[d * N + indexN];
    }
    if (d == 0)
    {
        DISTS  [indexB] = DISTS  [indexN];
        INDICES[indexB] = INDICES[indexN];
    }
}


__global__ static void computeDistancesToBForPLargerThanB_CUDA(int N, int D, int indexB, int numElements, FLOAT_TYPE *X, FLOAT_TYPE *DISTS, int *INDICES)
{
    int numThreads = blockDim.x;
    extern __shared__ FLOAT_TYPE array[];
    FLOAT_TYPE* s_posB = (FLOAT_TYPE*) array;
    int t = numThreads * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    FLOAT_TYPE dist = 0.0;

    unsigned int offsetD = 0;
    while (offsetD < D)
    {
        if (offsetD + tid < D) s_posB[tid] = X[(offsetD + tid) * N + indexB];
        __syncthreads();
        for (unsigned int d = offsetD; d < min(offsetD + numThreads, D); d++)
        {
            if (t < numElements) dist += distanceComponentGPU(X + d * N + (indexB + 1) + t, s_posB + d - offsetD);
        }
        offsetD += numThreads;
        __syncthreads();
    }
    if (t < numElements) dist = distanceFinalizeGPU(1, &dist);

    if (t < numElements)
    {
        int indexP = (t + indexB + 1);
        if (dist < DISTS[indexP])
        {
            DISTS  [indexP] = dist;
            INDICES[indexP] = indexB;
        }
    }
}



__global__ static void recomputeMinDistanceForElementAt_j_CUDA(int N, int D, int indexJ, FLOAT_TYPE *X, FLOAT_TYPE *DISTS, int *INDICES, FLOAT_TYPE *REDUCED_DISTS, int *REDUCED_INDICES)
{
    int numThreads = blockDim.x;

    extern __shared__ FLOAT_TYPE array[];
    FLOAT_TYPE* s_dist  = (FLOAT_TYPE*) array;
    int*   s_index = (int*  ) &s_dist[numThreads];
    FLOAT_TYPE* s_posJ  = (FLOAT_TYPE*) &s_index[numThreads];

    int t = numThreads * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    s_dist[tid] = FLT_MAX;
    if (t < indexJ) s_dist[tid] = 0.0;

    unsigned int offsetD = 0;
    while (offsetD < D)
    {
        if (offsetD + tid < D) s_posJ[tid] = X[(offsetD + tid) * N + indexJ];
        __syncthreads();
        for (unsigned int d = offsetD; d < min(offsetD + numThreads, D); d++)
        {
            if (t < indexJ) s_dist[tid] += distanceComponentGPU(X + d * N + t, s_posJ + d - offsetD);
        }
        offsetD += numThreads;
        __syncthreads();
    }
    if (t < indexJ)
    {
        s_dist[tid] = distanceFinalizeGPU(1, s_dist + tid);
        s_index[tid] = t;
    }
    __syncthreads();
    reduceMinTwo<THREADSPERBLOCK>(tid, s_dist, s_index);

    if (tid == 0)
    {
        REDUCED_DISTS  [blockIdx.x] = s_dist[tid];
        REDUCED_INDICES[blockIdx.x] = s_index[tid];
    }
}



int* hierarchicalGPU(int N, int D, FLOAT_TYPE *x)
{
    std::clock_t start;
    double duration;

    dim3 block(THREADSPERBLOCK);
    int numBlocks  = (int) ceil((FLOAT_TYPE) N / (FLOAT_TYPE) THREADSPERBLOCK);
    int numBlocksD = (int) ceil((FLOAT_TYPE) D / (FLOAT_TYPE) THREADSPERBLOCK);
    dim3 gridN(numBlocks);
    dim3 gridD(numBlocksD);
    int sMemReduce = (sizeof(FLOAT_TYPE) + sizeof(int)) * THREADSPERBLOCK;

    start = std::clock();
    std::cout<<"allocation started"<<std::endl ;
    FLOAT_TYPE *x_d             = allocDeviceMemory<FLOAT_TYPE*>(sizeof(FLOAT_TYPE) * D * N, x);
    int   *clustID_d       = allocDeviceMemory<int*>  (sizeof(int) * N);
    int   *closestClust_d  = allocDeviceMemory<int*>  (sizeof(int) * N);
    FLOAT_TYPE *closestDist_d   = allocDeviceMemory<FLOAT_TYPE*>(sizeof(FLOAT_TYPE) * N);

    FLOAT_TYPE *REDUCED_DISTS   = allocDeviceMemory<FLOAT_TYPE*>(sizeof(FLOAT_TYPE) * (numBlocks + 1));
    int   *REDUCED_INDICES = allocDeviceMemory<int*  >(sizeof(int) * (numBlocks + 1));
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    fprintf(stdout,"Device Memory allocation time:  %lf\n", duration);

    int* seq          = (int*) malloc(sizeof(int) * (N - 1) * 2);
    int* clustID      = (int*) malloc(sizeof(int) * N);
    int* closestClust = (int*) malloc(sizeof(int) * N);
    if (seq == NULL || clustID == NULL || closestClust == NULL)
    {
        cout << "Error in hierarchicalCPU(): Unable to allocate sufficient memory" << endl;
        exit(1);
    }

    unsigned int posA, posB, last, nextID = N - 1;

    start = std::clock();
    calcDistanceMatrix_CUDA<<<gridN, block>>>(N, D, x_d, closestClust_d, closestDist_d);
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    fprintf(stdout,"calcDistMatrix:  %lf\n", duration);
    start = std::clock();
    cudaMemcpy(closestClust, closestClust_d, sizeof(int) * N, cudaMemcpyDeviceToHost);


    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    fprintf(stdout,"cudaMemcpyDeviceToHost time:  %lf\n", duration);

    start = std::clock();
    for (int i = 0; i < N; i++) clustID[i] = i;
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    fprintf(stdout,"initialize clustID time:  %lf\n", duration);

    last = N;
    std::clock_t startTotalLoop = std::clock();
    for (int i = 0; i < N - 1; i++)
    {
        //fprintf(stdout,"Iteration:  %d\n", i);
        last--;
        nextID++;
        int newNumBlocks = (int) ceil((FLOAT_TYPE) last / (FLOAT_TYPE) THREADSPERBLOCK);
        dim3 newGridN(newNumBlocks);
        int sMem = sizeof(FLOAT_TYPE) * 2 * THREADSPERBLOCK + sizeof(int) * THREADSPERBLOCK;

        start = std::clock();
        cudaMemcpy(clustID_d, clustID, sizeof(int) * (last + 1), cudaMemcpyHostToDevice);
        duration = ( std::clock() - start ) / (double) (CLOCKS_PER_SEC/1000);
        //fprintf(stdout,"Step 1: memcopy host2device:  %lf\n", duration);

        // step1: get clustID for minimum distance
        start = std::clock();
        // posB = getMinDistIndex(last + 1, closestDist_d, closestClust_d, REDUCED_DISTS, REDUCED_INDICES);
        posB = getMinDistIndex(last + 1, closestDist_d, clustID_d, REDUCED_DISTS, REDUCED_INDICES);
        posA = closestClust[posB];

        seq[2 * i] = clustID[posA]; seq[2 * i + 1] = clustID[posB];
        duration = ( std::clock() - start ) / (double) (CLOCKS_PER_SEC/1000);
        //fprintf(stdout,"Step 1 getMinDistIndex:  %lf\n", duration);

        // step2: merge elements and insert at A, update distances to A as necessary
        start = std::clock();
        mergeElementsInsertAtA_CUDA<<<gridD, block>>>(N, D, posA, posB, x_d);
        duration = ( std::clock() - start ) / (double) (CLOCKS_PER_SEC/1000);
        //fprintf(stdout,"Step 2 mergeElementsInsertAtA_CUDA:  %lf\n", duration);

        clustID[posA] = nextID;

        if (posA != 0)
        {
            start = std::clock();
            computeAllDistancesToA_CUDA<<<newGridN, block, sMem>>>(N, last, D, posA, x_d, closestDist_d, closestClust_d, REDUCED_DISTS, REDUCED_INDICES);
            duration = ( std::clock() - start ) / (double) (CLOCKS_PER_SEC/1000);
            //fprintf(stdout,"Step 2 computeAllDistancesToA_CUDA:  %lf\n", duration);

            start = std::clock();
            //for positions less than indexA
            updateDistanceAndIndexForCluster(posA, sMemReduce, closestDist_d, closestClust_d, REDUCED_DISTS, REDUCED_INDICES);
            duration = ( std::clock() - start ) / (double) (CLOCKS_PER_SEC/1000);
            //fprintf(stdout,"Step 2 updateDistanceAndIndexForCluster:  %lf\n", duration);
        }

        // step3: replace cluster at B by last cluster
        start = std::clock();
        moveCluster_CUDA<<<gridD, block>>>(N, D, posB, last, x_d, closestDist_d, closestClust_d);
        duration = ( std::clock() - start ) / (double) (CLOCKS_PER_SEC/1000);
        //fprintf(stdout,"Step 3 moveCluster_CUDA:  %lf\n", duration);

        clustID[posB] = clustID[last];

        //keep if closestClust[last] is valid index, otherwise it will be covered in computeDistancesToBForPLargerThanB_CUDA
        if (closestClust[last] < posB) closestClust[posB] = closestClust[last];
        else closestClust[posB] = -1;

        if (last > posB)
        {
            dim3 gridLargerThanB((int) ceil((FLOAT_TYPE) (last - posB) / (FLOAT_TYPE) THREADSPERBLOCK)); // require (last - posB) threads
            int sMem2 = sizeof(FLOAT_TYPE) * THREADSPERBLOCK;

            start = std::clock();
            computeDistancesToBForPLargerThanB_CUDA<<<gridLargerThanB, block, sMem2>>>(N, D, posB, last - posB, x_d, closestDist_d, closestClust_d);
            duration = ( std::clock() - start ) / (double) (CLOCKS_PER_SEC/1000);
            //fprintf(stdout,"Step 3 computeDistancesToBForPLargerThanB_CUDA:  %lf\n", duration);

        }

        // step4: look for elements at positions > A that have A or B as nearest neighbor and recalculate distances if found
        // there is some redundancy possible, in case the new neighbor is the new A as this would already have been determined above
        start = std::clock();
        for (int j = posA + 1; j < last; j++)
        {
            int neighbor = closestClust[j];
            // Attention: uses original neighbor assignment; on device, for all elements that previously had element A as closest cluster, the neighbors have been set to -1
            if (neighbor == posA || neighbor == -1 || neighbor == posB)
            {
                int numBlocksJ = (int) ceil((FLOAT_TYPE) j / (FLOAT_TYPE) THREADSPERBLOCK);
                dim3 gridJ(numBlocksJ);
                recomputeMinDistanceForElementAt_j_CUDA<<<gridJ, block, sMem>>>(N, D, j, x_d, closestDist_d, closestClust_d, REDUCED_DISTS, REDUCED_INDICES);
                updateDistanceAndIndexForCluster(j, sMemReduce, closestDist_d, closestClust_d, REDUCED_DISTS, REDUCED_INDICES);
            }
        }
        duration = ( std::clock() - start ) / (double) (CLOCKS_PER_SEC/1000);
        fprintf(stdout,"Step 4 distance recalculation:  %lf\n", duration);
        cudaMemcpy(closestClust, closestClust_d, sizeof(int) * last, cudaMemcpyDeviceToHost);
    }
    duration = ( std::clock() - startTotalLoop ) / (double) (CLOCKS_PER_SEC/1000);
    fprintf(stdout,"total loop time:  %lf\n", duration);

    start = std::clock();
    cudaFree(x_d);
    cudaFree(clustID_d);
    cudaFree(closestClust_d);
    cudaFree(closestDist_d);
    cudaFree(REDUCED_DISTS);
    cudaFree(REDUCED_INDICES);
    free(clustID);
    free(closestClust);
    duration = ( std::clock() - start ) / (double) (CLOCKS_PER_SEC/1000);
    fprintf(stdout,"memory freeing time:  %lf\n", duration);

    return seq;
}
int N,K,D;
float* readFile(const char* fileName)
{
    float* data;
    string line;
    ifstream infile;
    float pars[3];
    int numData;
    FILE* f = fopen(fileName,"r");
    if (NULL == f) {
        printf("Failed to open data file");
        return NULL;
    }
    try
    {
        fscanf(f,"%f\n", &pars[0]);
        fscanf(f,"%f\n", &pars[1]);
        fscanf(f,"%f\n", &pars[2]);

        if (N == 0) N = (int) pars[0];
        if (K == 0) K = (int) pars[1];
        if (D == 0) D = (int) pars[2];

        numData = N * D;
        cout << "Reading " << numData << " floats" << endl;
        data = (float*) malloc(sizeof(float) * numData);
        memset(data, 0, sizeof(float) * numData);
        for (int i = 0; i < numData; i++)
        {
            fscanf(f,"%f\n", &data[i]);
        }
        cout << "Done reading" << numData << " floats" << endl;
    }
    catch (int e)
    {
        cout << "Error in dataIO::readFile(): ";
        if (e == 42) cout << "reached end of file \"" << fileName << "\" prematurely" << endl;
        else if (e == 1337) cout << "can only read floating point numbers" << endl;
        else cout << "reading file content failed" << endl;
        cout << "                             Please check parameters and file format" << endl;
        return NULL;
    }
    return data;
}

float* readData(const char* fileName)
{
    float* data;
    if (fileName == "") printf("filename not found");
    else data = readFile(fileName);
    return data;
}




int main(int argc, const char* argv[])
{
    //args: <data-file> <labels-file> <printEnabled : 1 to enable>
    std::clock_t start;
    double duration;

    start = std::clock();
    FLOAT_TYPE* x = readData(argv[1]);
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    fprintf(stdout,"reading float data:  %lf\n", duration);

    std::string labels[N];
    if(argc>=3)
    {
        std::ifstream file(argv[2]);
        std::string str;
        int i=0;
        while (std::getline(file, str))
        {
            // Process str
            labels[i++] = str;
            printf("%s\n",labels[i-1].c_str());
        }
    }
    const char *printEnabled;
    if(argc ==4)
        printEnabled = argv[3];

    printf("N = %d\n",N);
    printf("K = %d\n",K);
    printf("D = %d\n",D);

    start = std::clock();
    // do clustering on GPU
    int* seq = hierarchicalGPU(N, D, x);
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    fprintf(stdout,"clustering duration:  %lf\n", duration);
    free(x);

    int* ids = (int*) malloc(sizeof(int) * N);
    for (int i = 0; i < N; i++) ids[i] = i;
    if(atoi(printEnabled))
    {
    if (D < 3)
    {
        x = readData(argv[1]);
        // print all clusters (ATTENTION: only for D=1 and D=2)
        FLOAT_TYPE *num1 = (FLOAT_TYPE*) malloc(sizeof(FLOAT_TYPE) * D);
        FLOAT_TYPE *num2 = (FLOAT_TYPE*) malloc(sizeof(FLOAT_TYPE) * D);
        unsigned int pos1, pos2, id1, id2, nextID = N - 1;
        for (int i = 0; i < N - 1; i++)
        {
            nextID++;
            id1 = seq[2 * i]; id2 = seq[2 * i + 1];
            for (int j = 0; j < N; j++)
            {
                if (ids[j] == id1)
                {
                    for (unsigned int d = 0; d < D; d++) num1[d] = x[j + d * N];
                    pos1 = j;
                }
                if (ids[j] == id2)
                {
                    for (unsigned int d = 0; d < D; d++) num2[d] = x[j + d * N];
                    pos2 = j;
                }
            }
            cout << id1 << "\t" << id2 << "\t(";
            for (unsigned int d = 0; d < D; d++) cout << num1[d] << (d + 1 < D ? ", " : ") & (");
            for (unsigned int d = 0; d < D; d++) cout << num2[d] << (d + 1 < D ? ", " : ") => (");
            for (unsigned int d = 0; d < D; d++) cout << (num1[d] + num2[d]) / 2.0f << (d + 1 < D ? ", " : ") : ");
            cout << nextID << endl;
            for (unsigned int d = 0; d < D; d++)
            {
                x[d * N + pos1] = (x[d * N + pos1] + x[d * N + pos2]) / 2.0f;
            }
            ids[pos1] = nextID;
        }
        // free memory
        free(x);
    } else
    {
        unsigned int id1,id2,nextID=N-1;
        for (int i = 0; i < N - 1; i++) {
            nextID++;
            id1 = seq[2 * i]; id2 = seq[2 * i + 1];
            if(argc>=3)
            {
                std::string label1 = (id1<N)? labels[id1]: patch::to_string(id1);
                std::string label2 = (id2<N)? labels[id2]: patch::to_string(id2);
                cout << label1 << "\t" << label2 << "\t"<< nextID <<"\n";
            }else
            {
                cout << id1 << "\t" << id2 << "\t"<< nextID <<"\n";
            }
        }
    }
    }

    cout << "Done clustering" << endl;
    return 0;
}
