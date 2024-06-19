/**
 * \file learning.hh
 * Training and testing functions
*/


#ifndef LEARNING_HH
#define LEARNING_HH

#include "mlp.hh"
#include "activations.hh"
#include "top/net_ms.hh"
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

/**
 * Returns pixels vector of image
 * \param[in] fpath path to image
 * \return vector of pixels 
*/
std::vector<neuron_t> vectorize_image(std::string fpath);

/**
 * Returns label as a vector
 * \param[in] label label
 * \return vector as target neuron values 
*/
template<typename LabelType>
std::vector<neuron_t> vectorize_lable(LabelType& label);

/**
 * Train neural network (feedforward + back propagation)
 * \param[in] net neural network
 * \param[in] dir path, to training dataset
 * \param[in] learning_rate speed of neural network learning
*/
void train(NetMLP& net, std::string dir, double learning_rate);

/**
 * Test neural network 
 * \param[in] net neural network
 * \param[in] dir path, to testing dataset
 * \return total number of tests and how many of them passed successfully
*/
std::pair<size_t, size_t> test(NetMLP& net, std::string dir);


#endif