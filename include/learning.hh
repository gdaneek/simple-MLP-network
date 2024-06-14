
#ifndef LEARNING_HH
#define LEARNING_HH

#include "Graphics.hpp"
#include "include/mlp.hh"
#include "top/activations.hh"
#include "top/net_ms.hh"
#include "stdarg.h"

namespace fs = std::filesystem;
std::vector<neuron_t> vectorize_image(std::string fpath);

template <typename LabelType>
std::vector<neuron_t> vectorize_label(LabelType& label);
void train(NetMLP& net, std::string dir);
std::pair<size_t, size_t> test(NetMLP& net, std::string dir);

#endif
