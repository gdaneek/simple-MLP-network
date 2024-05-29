
#ifndef LEARNING_HH
#define LEARNING_HH

#include "../include/mlp.hh"
#include "../include/top/activations.hh"
#include "../include/top/net_ms.hh"
#include "stdarg.h"
#include <set>
#include <unordered_set>
#include <initializer_list>
#include <memory>
#include <map>
#include <ctime>
#include <iostream>
#include "../dependencies/SFML/include/Graphics.hpp"
#include <filesystem>       // ВЕРСИЯ CPP не ниже 17

namespace fs = std::filesystem;
vector_neuronval vectorize_image(std::string fpath);

template<typename LabelType>
vector_neuronval vectorize_label(LabelType& label);
void train(NetMLP& net, std::string dir);
std::tuple<size_t, size_t> test(NetMLP& net, std::string dir);


#endif