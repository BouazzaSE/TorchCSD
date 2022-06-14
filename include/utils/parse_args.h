#pragma once
#include <sstream>
#include <string>

namespace csd {
namespace utils {
namespace args {
template <typename T> inline std::string type_str();

template <> inline std::string type_str<bool>() { return "bool"; };

template <> inline std::string type_str<int>() { return "int"; };

template <> inline std::string type_str<long>() { return "long"; };

template <> inline std::string type_str<float>() { return "float"; };

template <> inline std::string type_str<std::string>() { return "std::string"; };

template <typename T> inline void parse_arg(int arg_idx, char *arg, T *out) {
  std::stringstream stream(arg);
  char c;
  T tmp_out;
  if (!((stream >> tmp_out) && !(stream >> c)))
    throw std::invalid_argument("Expected type " + type_str<T>() +
                                " for argument #" + std::to_string(arg_idx));
  *out = tmp_out;
}
} // namespace args
} // namespace utils
} // namespace csd
