
#include "top/activations.hh"

activations::Table activations::table = std::vector<activations::fptr>{
    new activations::Empty, new activations::ReLU,    new activations::Sigmoid,
    new activations::Tanh,  new activations::Softmax,
};