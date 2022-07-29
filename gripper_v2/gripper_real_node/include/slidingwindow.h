#ifndef SLIDINGWINDOW_H_
#define SLIDINGWINDOW_H_

#include <iostream>

namespace luke
{

template <typename T>
class SlidingWindow
{
  /* This class is a FIFO sliding window data structure */

public:

  std::vector<T> v;
  int i;

  SlidingWindow(int total_size) {
    // constructor
    if (total_size < 0) throw std::runtime_error("total size < 0");
    v.resize(total_size);
    i = -1;
  }

  void reset_to(const T& val) {
    v.assign(v.size(), val);
    i = -1;
  }

  void reset() {
    // reset the data to its default value
    reset_to(T{});
  }

  void add(T element) {
    // add an element
    i += 1;
    if (i > v.size() - 1) i = 0;
    v[i] = element;
  }

  T read_element() {
    // get the most recently added element
    if (i == -1) return v[0]; // if the window is empty, return T{}, default value
    return v[i];
  }

  T read_element(int n) {
    // get the element added n steps ago - beware looping
    int idx = i - n;
    while (idx < 0) idx += v.size();
    return v[idx];
  }

  std::vector<T> read(int n) {
    // read n elements in the window into a vector, most recent element last
    if (n < 0) throw std::runtime_error("cannot read with n < 0");
    std::vector<T> out;
    for (int j = n - 1; j >= 0; j--) {
      int idx = i - j;
      while (idx < 0) idx += v.size();
      out.push_back(v[idx]);
    }
    return out;
  }

  std::vector<T> read_backwards(int n) {
    // read n elements in the window into a vector, most recent element first
    if (n < 0) throw std::runtime_error("cannot read backwards with n < 0");
    std::vector<T> out;
    for (int j = 0; j < n; j++) {
      int idx = i - j;
      while (idx < 0) idx += v.size();
      out.push_back(v[idx]);
    }
    return out;
  }

  void print(int n = 0) {
    // print the data of an n element reading, most recent element last
    if (n < 0) throw std::runtime_error("cannot print with n < 0");
    if (n == 0) n = v.size();
    std::vector<T> data = read(n);
    std::cout << "Sliding window data reading (n = " << n << "): { ";
    for (int j = 0; j < data.size(); j++) {
      std::cout << data[j] << " ";
    }
    std::cout << "}\n";
  }

  void print_backwards(int n = 0) {
    // print the data of an n element reading, most recent element first
    if (n < 0) throw std::runtime_error("cannot print with n < 0");
    if (n == 0) n = v.size();
    std::vector<T> data = read_backwards(n);
    std::cout << "Sliding window data reading (n = " << n << "): { ";
    for (int j = 0; j < data.size(); j++) {
      std::cout << data[j] << " ";
    }
    std::cout << "}\n";
  }

};

} // namespace luke

#endif