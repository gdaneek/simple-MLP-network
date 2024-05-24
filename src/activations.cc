#include "../include/top/activations.hh"

#include <iostream>
#include <stdexcept>
double activations::derivative(Neuron& neuron, activations::flink f, double accuracy) {
    neuron += accuracy;
    f(neuron);
    auto nval = neuron.get_value();
    neuron += -accuracy;
    return (nval - neuron.get_value()) / accuracy;
}

void activations::Empty::operator ()(Neuron& neuron) {}

void activations::Empty::operator ()(Layer& layer) {}

std::string activations::Empty::name() {
    return "empty";
}

void  activations::ReLU::operator ()(Neuron& neuron) {
     if(neuron.get_value() < 0)
        neuron = 0; 

}
void  activations::ReLU::operator ()(Layer& layer)   {
    //for(Neuron& neuron : layer)(*this)(neuron);
    for(size_t i{0};i < layer.size();(*this)(layer[i++]));
}

std::string activations::ReLU::name() {
    return "ReLU";
}

void activations::Tanh::operator ()(Neuron& neuron) {
    neuron = std::tanh(neuron.get_value());
}

void activations::Tanh::operator ()(Layer& layer)  {
    //for(Neuron& neuron : layer)(*this)(neuron);
    for(size_t i{0};i < layer.size();(*this)(layer[i++]));
}

std::string activations::Tanh::name(){
    return "tanh";
}

void activations::Sigmoid::operator ()(Neuron& neuron) {
    neuron = 1.0/(1.0+std::exp(-1.0*neuron.get_value())); 
}

void activations::Sigmoid::operator ()(Layer& layer)  {
   // for(Neuron& neuron : layer)(*this)(neuron);
   for(size_t i{0};i < layer.size();(*this)(layer[i++]));
}

std::string activations::Sigmoid::name(){
    return "sigmoid";
}

void  activations::Softmax::operator ()(Neuron& neuron) {
    neuron = std::exp(neuron.get_value())/exp_sum;
}

void activations::Softmax::operator ()(Layer& layer)    {
    exp_sum = 0;
    //for(Neuron& neuron : layer)exp_sum += std::exp(neuron.get_value());
    //for(Neuron& neuron : layer)(*this)(neuron);
    for(size_t i{0};i < layer.size();exp_sum += std::exp(layer[i++].get_value()));
    for(size_t i{0};i < layer.size();(*this)(layer[i++]));

}

std::string activations::Softmax::name() {
    return "softmax";
}

size_t activations::Table::make_id(activations::fname name) {
    return (size_t)std::hash<std::string>{}(name);
}

activations::fptr activations::Table::get_by_id(size_t id) {
    for(auto af : table)
        if(activations::Table::make_id(af->name()) == id)
            return af;
    throw std::runtime_error{"Requested activation function with id "+std::to_string(id)+" does not exist"};
    return nullptr;
}

activations::fptr activations::Table::get_by_name(activations::fname name) {
    for(auto af : table)
        if(af->name() == name)
            return af;
    throw std::runtime_error{"Requested activation function with name "+name+" does not exist"};
    return nullptr;
}

activations::fname activations::Table::name_by_id(size_t id) {
    for(auto af : table)
        if(activations::Table::make_id(af->name()) == id)
            return af->name();
    return "undefined";
}

activations::fname activations::Table::name_by_ptr(activations::fptr ptr) { 
    return ptr->name();
}

size_t activations::Table::id_by_ptr(activations::fptr ptr) {
    return make_id(ptr->name());
}