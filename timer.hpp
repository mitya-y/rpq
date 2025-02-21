#include <time.h>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>

class Timer {
private:
  using TimeT = struct timespec;
  TimeT start {}, finish {};

public:
  Timer() { mark(); }

  void mark() { clock_gettime(CLOCK_MONOTONIC, &start); }

  double measure() {
    clock_gettime(CLOCK_MONOTONIC, &finish);
    double elapsed =
      (finish.tv_sec - start.tv_sec) * 1000000.0 + (finish.tv_nsec - start.tv_nsec) / 1000.0;
    elapsed /= 1'000'000;  // translate to sec
    // std::cout << msg << elapsed << "s\n";
    mark();
    return elapsed;
  }
};

inline static std::string execute_command(const char* cmd) {
  auto deleter = [](FILE *f) {
    if (f != nullptr) {
      pclose(f);
    }
  };
  std::unique_ptr<FILE, decltype(deleter)> pipe(popen(cmd, "r"), deleter);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }

  std::array<char, 128> buffer;
  std::string result;
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
