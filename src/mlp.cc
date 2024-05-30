#include "../include/mlp.hh"
#include </usr/include/SFML/Graphics.hpp>

NeuralLink::NeuralLink(size_t input_layer_size, size_t output_layer_size) {
    weights = weight_matrix(input_layer_size, weight_vector(output_layer_size));
    for(size_t i{0};i < input_layer_size;i++)
        for(size_t j{0};j < output_layer_size;j++)
            weights[i][j] = (weight)(rand())/RAND_MAX;
}
size_t NeuralLink::size() {
    return weights.size();
}

weight_vector& NeuralLink::operator [](int64_t i) {
    return weights[(i < 0)? weights.size()+i : i];  
}

weight_matrix::iterator NeuralLink::begin() {
    return weights.begin();
}

weight_matrix::iterator NeuralLink::end() {
    return weights.end();
}

weight_matrix::reverse_iterator NeuralLink::rbegin() {
    return weights.rbegin();
}

weight_matrix::reverse_iterator NeuralLink::rend() {
    return weights.rend();
}

NeuronMLP::NeuronMLP() {
    value = 0;
    shift = (neuronval)(rand())/RAND_MAX;
}

NeuronMLP::NeuronMLP(NeuronMLP&& other){
    value = other.value;
    shift = other.shift;
}
NeuronMLP::NeuronMLP(NeuronMLP const& other){
    value = other.value;
    shift = other.shift;
}

NeuronMLP::NeuronMLP(const neuronval _value) {
        shift = (neuronval)(rand())/RAND_MAX;
        value = _value;
}

constexpr NeuronMLP& NeuronMLP::operator=(const NeuronMLP& other) {
    value = other.value;
    shift = other.shift;
    return *this;
}

Neuron& NeuronMLP::operator =(const neuronval value) {
        this->value = value;
        return *this;
}

Neuron& NeuronMLP::operator +=(const neuronval value) {
    this->value += value;
    return *this;
}

Neuron& NeuronMLP::operator *=(const neuronval value) {
    this->value *= value;
    return *this;
}

NeuronMLP::operator neuronval() const {
    return value;
}
LayerMLP::LayerMLP(size_t size) {
    neurons.resize(size, NeuronMLP(0));
}

LayerMLP::LayerMLP(std::vector<NeuronMLP>& neurons) {
    this->neurons = neurons;
}

LayerMLP::LayerMLP(vector_neuronval& neuron_values) {
    neurons.clear();
    for(auto value : neuron_values)
        neurons.push_back(NeuronMLP(value));
}                                      
LayerMLP::LayerMLP(vector_neuronval&& neuron_values) {
    neurons.clear();
    for(auto value : neuron_values)
        neurons.push_back(NeuronMLP(value));
}          
LayerMLP& LayerMLP::operator =(vector_neuronval&& values) {
    (*this) = LayerMLP(values); 
    return *this;
}

LayerMLP::LayerMLP(const LayerMLP& other) {
    neurons = other.neurons;
}                                    

void LayerMLP::apply_offsets() {
    for(NeuronMLP& neuron : neurons)
        neuron += neuron.shift;
}

size_t LayerMLP::size() const {
    return neurons.size();
}

NeuronMLP& LayerMLP::operator [](int64_t i) {
    return neurons[(i < 0)? neurons.size()+i : i];
}

LayerMLP& LayerMLP::operator=(LayerMLP&& layer) {
     if(layer.size() != neurons.size())
        throw std::runtime_error{"The size of the layer and the size of the other layer do not match"};
    for(size_t i{0}; i < layer.size();neurons[i] = layer[i], i++);
    return *this;  
}
LayerMLP& LayerMLP::operator=(const LayerMLP& other){
    return this->operator=(std::move(other));
}

LayerIteratorProxy LayerMLP::begin() {
    return LayerIteratorProxy(new LayerMLP::iterator{&*neurons.begin()});
};
LayerIteratorProxy LayerMLP::end() {
    return LayerIteratorProxy(new LayerMLP::iterator{&*neurons.end()});
};           
Layer::iterator& LayerMLP::iterator::operator--(int) {
    LayerMLP::iterator it = *this;
    --*this;
    return it;

}
Layer::iterator& LayerMLP::iterator::operator++(int) {
    LayerMLP::iterator it = *this;
    ++*this;
    return it;
}
Layer::iterator& LayerMLP::iterator::operator++() {
    ptr = static_cast<NeuronMLP*>(ptr)+1;
    return *this;
}
Layer::iterator&LayerMLP::iterator::operator--(){
    ptr = static_cast<NeuronMLP*>(ptr)-1;
    return *this;

}
Neuron& LayerMLP::iterator::operator*() {
    return *ptr;
}
Neuron* LayerMLP::iterator::operator->() {
    return ptr;
}
bool LayerMLP::iterator::operator==(Layer::iterator& other) {
    return ptr == other.ptr;
}
bool LayerMLP::iterator::operator!=(Layer::iterator& other) {
    return ptr != other.ptr;
}

void NetMLP::add(size_t neuron_count, activations::fptr activation, size_t layer_index) {
   table.insert({((layer_index)? layer_index : ++layer_indexer), neuron_count, activation});
}

NetMLP::NetMLP(size_t in_sz, activations::fptr in, size_t out_sz, activations::fptr out) {
    layer_indexer = 0;  
    add(in_sz, in, ++layer_indexer);
    add(out_sz, out, -1);
}

void NetMLP::make() {
    if(!table.size())
        return;
    layers.clear(); links.clear(); activations.clear();
    size_t prev_layer_size{0};
    std::unordered_set<size_t> indexes;
    for(auto& row : table) {
        if(indexes.count(std::get<0>(row)))
            throw std::runtime_error{"Network creation error: several different layers have the same index"};
        else 
            indexes.insert(std::get<0>(row));
        layers.push_back(std::get<1>(row));
        activations.push_back(std::get<2>(row));
        if(prev_layer_size) 
            links.push_back(NeuralLink(prev_layer_size, (*--layers.end()).size()));
        prev_layer_size = (*--layers.end()).size();
    }
    table.clear();
}

size_t NetMLP::size() const {
    return layers.size();
}

NetMLP::NetMLP(NetMLP&& other) {
    table.clear();
    layers = other.layers;
    links = other.links;
    activations = other.activations;
}
NetMLP::NetMLP(const NetMLP& other) {
    table.clear();
    layers = other.layers;
    links = other.links;
    activations = other.activations;
}

LayerMLP operator*(LayerMLP& layer, NeuralLink& links) {
    weight_vector weights(links[0].size(), 0);
    for(size_t neuron_it{0};neuron_it < layer.size();neuron_it++) 
        for(size_t weight_it{0};weight_it < links[neuron_it].size();weight_it++) 
            weights[weight_it] += layer[neuron_it].get_value() * links[neuron_it][weight_it];
    return LayerMLP(weights);
}

void NetMLP::set_input(weight_vector& values) {
    if(enable_automake && table.size())make();
    if(!layers.size())
        throw std::runtime_error{"It's impossible to calculate the result for an unmaked net"};
    layers[0] = values;
}
void NetMLP::calc_output() {
    if(enable_automake && table.size())make();
    for(size_t i{1};i < layers.size();i++) {
        layers[i] = layers[i-1]*links[i-1];
        layers[i].apply_offsets();
        (*activations[i])(layers[i]);
    }
}
//void NetMLP::tng(){backprop();}
void NetMLP::backprop(const std::vector<double>& target, double learning_rate) {
    size_t last = layers.size() - 1;
    weight_vector delta_previous(layers.size());
    for (size_t j{0}; j < layers.back().size(); j++) {
        delta_previous[j] = -2 * learning_rate * layers.back()[j].get_value() * (1 - layers.back()[j].get_value()) * (target[j] - layers.back()[j].get_value());         // j ????
    }
    weight_vector delta;
    for (size_t j{last - 1}; j >= 0; j--) {
        delta.resize(layers[j].size());
        // find d
        for (size_t k{0}; k < layers[j].size(); k++) {
            double sum_d = 0;
            for (size_t t{0}; t < delta_previous.size(); t++) {
                // sum_d += delta_previous[t] * net.NeuralLinks[j].weights[k][t];
                // sum_d += delta_previous[t] * links[j][k][t];                 same???
                sum_d += delta_previous[t] * links[j].weights[k][t];
            }
            delta[k] = 2 * learning_rate * layers[k][j].get_value() * (1 - layers[k][j].get_value()) * sum_d;
        }
        // change w with delta_previous (from k to t) 
        for (size_t k{0}; k < layers[j].size(); k++) {
            layers[j][k].shift -= delta_previous[k];                       // shift changes
            for (size_t t{0}; t < links.size(); t++) {
                double dw = delta_previous[t] * layers[j][k].get_value(); 
                links[j].weights[k][t] += dw;
            }
        }
        // delta_previous = delta
        delta_previous.clear();
        delta_previous.resize(delta.size());
        delta_previous = delta;
        delta.clear();
    }
}

std::tuple<size_t, neuronval> NetMLP::response() {      
    std::tuple<size_t, neuronval> neuron {0, (*--layers.end())[0].get_value()};
    for(size_t i{0};i < (*--layers.end()).size();i++) 
        if(std::get<neuronval>(neuron) < (*--layers.end())[i].get_value()) 
            neuron = {i, (*--layers.end())[i].get_value()};
    return neuron;
}
std::tuple<size_t, neuronval> NetMLP::feedforward(vector_neuronval& input_values) {
    set_input(input_values);
    calc_output();
    return response();
}
void NetMLP::feedforward() {
    calc_output();
}

void NetMLP::disable_automake() {   // Отключает автосборку сети, есть таблица непустая
    enable_automake = false;
}