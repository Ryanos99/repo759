#include <cstdio>
#include <cuda_runtime.h>
#include <random>

__global__ void axpyKernel(int *dA, int a) {
  int x = threadIdx.x;
  int y = blockIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  dA[idx] = a * x + y;
}

int main() {
  const int numBlocks = 2;
  const int threadsPerBlock = 8;
  const int n = numBlocks * threadsPerBlock;

  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<int> dist(1, 10);
  int a = dist(generator);

  int *dA = nullptr;
  cudaMalloc((void **)&dA, n * sizeof(int));

  axpyKernel<<<numBlocks, threadsPerBlock>>>(dA, a);

  int hA[n];
  cudaMemcpy(hA, dA, n * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < n; ++i) {
    std::printf("%d%s", hA[i], i < n - 1 ? " " : "\n");
  }

  cudaFree(dA);
  return 0;
}
