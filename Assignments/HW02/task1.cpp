#include "scan.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <ratio>

using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

float randomFloat() {
  static std::mt19937 gen(std::random_device{}());
  static std::uniform_real_distribution<float> dist(-1.0f,
                                                    std::nextafter(1.0f, 2.0f));
  return dist(gen);
}

int main(int, char *argv[]) {
  int n{std::atoi(argv[1])};

  float *arr{static_cast<float *>(std::malloc(n * sizeof(float)))};
  float *output{static_cast<float *>(std::malloc(n * sizeof(float)))};

  for (int i{}; i < n; i++) {
    arr[i] = randomFloat();
  }

  auto start = high_resolution_clock::now();

  scan(arr, output, n);

  auto end = high_resolution_clock::now();

  auto duration_ms =
      std::chrono::duration_cast<duration<double, std::milli>>(end - start);

  cout << duration_ms.count() << std::endl;
  cout << output[0] << std::endl;
  cout << output[n - 1] << std::endl;

  std::free(arr);
  std::free(output);
}
