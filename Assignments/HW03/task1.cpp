#include "matmul.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <ratio>
#include <vector>

using std::cout;
using std::chrono::duration;
using std::chrono::high_resolution_clock;

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " n t\n";
    return 1;
  }

  const std::size_t n = std::strtoull(argv[1], nullptr, 10);
  const int t = std::atoi(argv[2]);
  const std::size_t size = n * n;
  std::vector<float> A(size), B(size), C(size, 0.0f);

 
  std::random_device entropy_source;
  std::mt19937 generator(entropy_source());
  std::uniform_real_distribution<float> dist(
      -1.0f, std::nextafter(1.0f, std::numeric_limits<float>::max()));
  for (std::size_t i = 0; i < size; ++i) {
    A[i] = dist(generator);
    B[i] = dist(generator);
  }

  omp_set_num_threads(t);

  high_resolution_clock::time_point start;
  high_resolution_clock::time_point end;
  duration<double, std::milli> duration_ms;

  start = high_resolution_clock::now();
  mmul(A.data(), B.data(), C.data(), n);
  end = high_resolution_clock::now();

  duration_ms = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

  cout << C.front() << '\n' << C.back() << '\n' << duration_ms.count() << '\n';
  return 0;
}
