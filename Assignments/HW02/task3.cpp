#include "matmul.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <ratio>
#include <vector>

int main() {
  const unsigned int n = 1024;
  const std::size_t size = static_cast<std::size_t>(n) * n;
  std::vector<double> A(size), B(size), C(size);

  std::mt19937 generator(std::random_device{}());
  std::uniform_real_distribution<double> values(-1.0, 1.0);
  for (std::size_t i = 0; i < size; ++i) {
    A[i] = values(generator);
    B[i] = values(generator);
  }

  std::cout << n << '\n';

  for (auto multiply : {mmul1, mmul2, mmul3}) {
    std::fill(C.begin(), C.end(), 0.0);
    const auto start = std::chrono::high_resolution_clock::now();
    multiply(A.data(), B.data(), C.data(), n);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            end - start);

    std::cout << elapsed.count() << '\n' << C.back() << '\n';
  }

  std::fill(C.begin(), C.end(), 0.0);
  const auto start = std::chrono::high_resolution_clock::now();
  mmul4(A, B, C.data(), n);
  const auto end = std::chrono::high_resolution_clock::now();
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          end - start);

  std::cout << elapsed.count() << '\n' << C.back() << '\n';
  return 0;
}
