
#ifndef LEARNING_HH
#define LEARNING_HH

#include "mlp.hh"
#include "activations.hh"
#include "net_ms.hh"
#include "stdarg.h"
#include <set>
#include <unordered_set>
#include <initializer_list>
#include <memory>
#include <map>
#include <ctime>
#include <iostream>
#include "Graphics.hpp"
#include <filesystem>       // ВЕРСИЯ CPP не ниже 17

namespace fs = std::filesystem;
std::vector<neuron_t> vectorize_image(std::string fpath);

template<typename LabelType>
std::vector<neuron_t> vectorize_label(LabelType& label);
void train(NetMLP& net, std::string dir);
std::tuple<size_t, size_t> test(NetMLP& net, std::string dir);


#endif