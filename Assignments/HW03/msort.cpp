#include "msort.h"

#include <vector>

namespace {

void merge(int *arr, int *tmp, const std::size_t mid, const std::size_t n) {
  std::size_t i = 0;
  std::size_t j = mid;
  std::size_t k = 0;
  while (i < mid && j < n) {
    tmp[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
  }
  while (i < mid) {
    tmp[k++] = arr[i++];
  }
  while (j < n) {
    tmp[k++] = arr[j++];
  }
  for (k = 0; k < n; ++k) {
    arr[k] = tmp[k];
  }
}

void serial_sort(int *arr, int *tmp, const std::size_t n) {
  for (std::size_t width = 1; width < n; width *= 2) {
    for (std::size_t lo = 0; lo + width < n; lo += 2 * width) {
      const std::size_t len = (n - lo < 2 * width) ? n - lo : 2 * width;
      merge(arr + lo, tmp + lo, width, len);
    }
  }
}

void parallel_sort(int *arr, int *tmp, const std::size_t n,
                   const std::size_t threshold) {
  if (n < threshold || n < 2) {
    serial_sort(arr, tmp, n);
    return;
  }
  const std::size_t mid = n / 2;
#pragma omp task
  parallel_sort(arr, tmp, mid, threshold);
#pragma omp task
  parallel_sort(arr + mid, tmp + mid, n - mid, threshold);
#pragma omp taskwait
  merge(arr, tmp, mid, n);
}

}  

void msort(int *arr, const std::size_t n, const std::size_t threshold) {
  if (n < 2) {
    return;
  }
  std::vector<int> tmp(n);

  if (omp_in_parallel()) {
    parallel_sort(arr, tmp.data(), n, threshold);
  } else {
#pragma omp parallel
#pragma omp single
    parallel_sort(arr, tmp.data(), n, threshold);
  }
}
