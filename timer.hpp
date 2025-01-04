#include <time.h>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

class Timer {
private:
  inline static struct timespec start {}, finish {};

public:
  static void mark() { clock_gettime(CLOCK_MONOTONIC, &start); }

  static double measure() {
    clock_gettime(CLOCK_MONOTONIC, &finish);
    double elapsed =
      (finish.tv_sec - start.tv_sec) * 1000000.0 + (finish.tv_nsec - start.tv_nsec) / 1000.0;
    elapsed /= 1'000'000;  // translate to sec
    // std::cout << msg << elapsed << "s\n";
    mark();
    return elapsed;
  }
};

inline static std::string exec(const char* cmd) {
  std::array<char, 128> buffer;
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  return result;
}

inline static uint32_t parse_int(const std::string& s) {
  uint32_t value;
  std::from_chars(s.c_str(), s.c_str() + s.size(), value);
  return value;
}
