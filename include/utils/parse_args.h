/*
Copyright 2022 Bouazza SAADEDDINE

This file is part of TorchCSD.

TorchCSD is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

TorchCSD is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with TorchCSD.  If not, see <https://www.gnu.org/licenses/>.
*/



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
