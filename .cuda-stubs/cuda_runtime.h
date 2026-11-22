// Editor-only stand-in for the CUDA runtime header so clangd can parse .cu
// files on macOS (where the CUDA toolkit is unavailable). Never used by nvcc:
// on Euler the real <cuda_runtime.h> is found first.
#pragma once

#include <cstddef>
#include <cstdio>

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __managed__ __attribute__((managed))
#define __forceinline__ inline __attribute__((always_inline))
#define __restrict__ __restrict

// ---- vector / index types ----
struct uint3 {
  unsigned int x, y, z;
};
struct dim3 {
  unsigned int x, y, z;
  __host__ __device__ constexpr dim3(unsigned int vx = 1, unsigned int vy = 1,
                                     unsigned int vz = 1)
      : x(vx), y(vy), z(vz) {}
  __host__ __device__ constexpr dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
};

extern __device__ const uint3 threadIdx;
extern __device__ const uint3 blockIdx;
extern __device__ const dim3 blockDim;
extern __device__ const dim3 gridDim;
extern __device__ const int warpSize;

// ---- runtime API types ----
enum cudaError_t { cudaSuccess = 0, cudaErrorUnknown = 999 };
typedef cudaError_t cudaError;
enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4
};
typedef struct CUevent_st *cudaEvent_t;
typedef struct CUstream_st *cudaStream_t;

// Needed by clang to lower the <<<grid, block, shmem, stream>>> syntax
// (which one it looks for depends on the assumed CUDA version).
extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                         size_t sharedMem = 0,
                                         cudaStream_t stream = 0);
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                                size_t sharedMem = 0,
                                                cudaStream_t stream = 0);

// ---- runtime API functions ----
cudaError_t cudaMalloc(void **devPtr, size_t size);
template <class T> cudaError_t cudaMalloc(T **devPtr, size_t size) {
  return cudaMalloc((void **)devPtr, size);
}
cudaError_t cudaMallocManaged(void **devPtr, size_t size,
                              unsigned int flags = 1);
template <class T>
cudaError_t cudaMallocManaged(T **devPtr, size_t size, unsigned int flags = 1) {
  return cudaMallocManaged((void **)devPtr, size, flags);
}
cudaError_t cudaMallocHost(void **ptr, size_t size);
cudaError_t cudaFree(void *devPtr);
cudaError_t cudaFreeHost(void *ptr);
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       cudaMemcpyKind kind);
cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src,
                               size_t count, size_t offset = 0,
                               cudaMemcpyKind kind = cudaMemcpyHostToDevice);
cudaError_t cudaMemset(void *devPtr, int value, size_t count);
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaDeviceReset();
cudaError_t cudaGetLastError();
cudaError_t cudaPeekAtLastError();
const char *cudaGetErrorString(cudaError_t error);
cudaError_t cudaEventCreate(cudaEvent_t *event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);
cudaError_t cudaEventDestroy(cudaEvent_t event);

// ---- device intrinsics ----
__device__ void __syncthreads();
__device__ int __syncthreads_count(int predicate);
__device__ void __threadfence();
__device__ int atomicAdd(int *address, int val);
__device__ unsigned int atomicAdd(unsigned int *address, unsigned int val);
__device__ float atomicAdd(float *address, float val);
__device__ double atomicAdd(double *address, double val);
__device__ int atomicMax(int *address, int val);
__device__ int atomicMin(int *address, int val);
__device__ int atomicExch(int *address, int val);
__device__ int atomicCAS(int *address, int compare, int val);
__device__ float __shfl_down_sync(unsigned mask, float var, unsigned delta,
                                  int width = 32);
__device__ int __shfl_down_sync(unsigned mask, int var, unsigned delta,
                                int width = 32);
