#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>
#include <vector>

#include "vscale.cuh"

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 512
#endif

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::fprintf(stderr, "Usage: %s n\n", argv[0]);
    return 1;
  }
  const unsigned int n = std::strtoul(argv[1], nullptr, 10);
  if (n == 0) {
    std::fprintf(stderr, "n must be a positive integer\n");
    return 1;
  }
  const unsigned int threadsPerBlock = THREADS_PER_BLOCK;

  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_real_distribution<float> distA(-10.0f, 10.0f);
  std::uniform_real_distribution<float> distB(0.0f, 1.0f);

  std::vector<float> hA(n), hB(n);
  for (unsigned int i = 0; i < n; ++i) {
    hA[i] = distA(generator);
    hB[i] = distB(generator);
  }

  float *dA = nullptr, *dB = nullptr;
  cudaMalloc((void **)&dA, n * sizeof(float));
  cudaMalloc((void **)&dB, n * sizeof(float));
  cudaMemcpy(dA, hA.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  const unsigned int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  vscale<<<numBlocks, threadsPerBlock>>>(dA, dB, n);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);

  cudaMemcpy(hB.data(), dB, n * sizeof(float), cudaMemcpyDeviceToHost);

  std::printf("%f\n%f\n%f\n", ms, hB[0], hB[n - 1]);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(dA);
  cudaFree(dB);
  return 0;
}
