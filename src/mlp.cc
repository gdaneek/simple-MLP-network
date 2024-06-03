#include "../include/mlp.hh"

MLPLink::MLPLink(const size_t input_layer_size,const size_t output_layer_size) {
    if(!input_layer_size || !output_layer_size)
        throw std::runtime_error{"trying to create links for a layer with size 0"};
    rows = input_layer_size;
    cols = output_layer_size;
    weights = weight_vector(rows*cols);
    for(size_t i{0};i < rows*cols;weights[i] = (weight)(rand())/RAND_MAX,i++);
}

size_t MLPLink::size() const {
    return weights.size();
}

size_t MLPLink::get_cols() const {
    return cols;
};

size_t MLPLink::get_rows() const {
    return rows;
}

size_t MLPLink::input_layer_size() const {
    return get_rows();
};

size_t MLPLink::output_layer_size() const {
    return get_cols();
};

weight& MLPLink::get_weight(const int64_t i, const int64_t j) { // вес между i-тым нейроном текущего слоя и j-тым предыдущего
    return weights[cols*j + i];
}

 NLinkIteratorProxy MLPLink::begin() {
    return NLinkIteratorProxy(new MLPLink::iterator{&*weights.begin()});
}

NLinkIteratorProxy MLPLink::end() {
    return NLinkIteratorProxy(new MLPLink::iterator{&*weights.end()});
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-local-addr"

NLink::iterator& MLPLink::iterator::operator--(int) {
    MLPLink::iterator it = *this;
    --*this;
    return it;
}
NLink::iterator& MLPLink::iterator::operator++(int) {
    MLPLink::iterator it = *this;
    ++*this;
    return it;
}
NLink::iterator& MLPLink::iterator::operator++() {
    ptr = ptr+1;
    return *this;
}
NLink::iterator&MLPLink::iterator::operator--(){
    ptr = ptr-1;
    return *this;

}
weight& MLPLink::iterator::operator*() {
    return *ptr;
}
weight* MLPLink::iterator::operator->() {
    return ptr;
}
bool MLPLink::iterator::operator==(NLink::iterator& other) {
    return ptr == other.ptr;
}
bool MLPLink::iterator::operator!=(NLink::iterator& other) {
    return ptr != other.ptr;
}
#pragma GCC diagnostic pop

NeuronMLP::NeuronMLP() {
    value = 0;
    shift = (neuron_t)(rand())/RAND_MAX;
}

NeuronMLP::NeuronMLP(NeuronMLP&& other){
    value = other.value;
    shift = other.shift;
}
NeuronMLP::NeuronMLP(NeuronMLP const& other){
    value = other.value;
    shift = other.shift;
}

NeuronMLP::NeuronMLP(const neuron_t _value) {
        shift = (neuron_t)(rand())/RAND_MAX;
        value = _value;
}

const NeuronMLP& NeuronMLP::operator=(const NeuronMLP& other) {
    value = other.value;
    shift = other.shift;
    return *this;
}

Neuron& NeuronMLP::operator =(const neuron_t value) {
        this->value = value;
        return *this;
}

Neuron& NeuronMLP::operator +=(const neuron_t value) {
    this->value += value;
    return *this;
}

Neuron& NeuronMLP::operator *=(const neuron_t value) {
    this->value *= value;
    return *this;
}

NeuronMLP::operator neuron_t() const {
    return value;
}
LayerMLP::LayerMLP(const size_t size) {
    if(!size)
        throw std::runtime_error{"Layer size cannot be zero"};
    neurons.resize(size, NeuronMLP(0));
}

LayerMLP::LayerMLP(std::vector<NeuronMLP> const& neurons) {
    if(!neurons.size())
        throw std::runtime_error{"Layer size cannot be zero"};
    this->neurons = neurons;
}

LayerMLP::LayerMLP(std::vector<neuron_t> const& neuron_values) {
    neurons.clear();
    for(auto value : neuron_values)
        neurons.push_back(NeuronMLP(value));
}                                      
LayerMLP::LayerMLP(std::vector<neuron_t>&& neuron_values) {
    neurons.clear();
    for(auto value : neuron_values)
        neurons.push_back(NeuronMLP(value));
}          

LayerMLP& LayerMLP::operator =(std::vector<neuron_t>&& values) {
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

LayerMLP& LayerMLP::operator=(LayerMLP&& other) {
     return this->operator=(other);
    
}
const NeuronMLP LayerMLP::get_neuron(int64_t i) const {return neurons[(i < 0)? neurons.size()+i : i];}
LayerMLP& LayerMLP::operator=(const LayerMLP& other){
    if(other.size() != neurons.size())
        throw std::runtime_error{"The size of the layer and the size of the other layer do not match"};
    for(size_t i{0}; i < other.size();neurons[i] = (NeuronMLP)other.get_neuron(i), i++);
    return *this;  
}

LayerIteratorProxy LayerMLP::begin() {
    return LayerIteratorProxy(new LayerMLP::iterator{&*neurons.begin()});
};
LayerIteratorProxy LayerMLP::end() {
    return LayerIteratorProxy(new LayerMLP::iterator{&*neurons.end()});
};           

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-local-addr"


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

#pragma GCC diagnostic pop

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
            links.push_back(MLPLink(prev_layer_size, (*--layers.end()).size()));
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

LayerMLP operator*(LayerMLP& layer, MLPLink& link) {
    weight_vector weights(link.output_layer_size(), 0);
    if(layer.size() != link.input_layer_size())
        throw std::runtime_error{"Multiplication cannot be performed: the layer size and the number of rows in nlinks do not match"};
    for(size_t neuron_it{0};neuron_it < layer.size();neuron_it++) 
        for(size_t weight_it{0};weight_it < link.output_layer_size();weight_it++) 
            weights[weight_it] += layer[neuron_it].get_value() * link.get_weight(weight_it, neuron_it);
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
void NetMLP::backprop(const std::vector<double>& target, double learning_rate) {}

std::tuple<size_t, neuron_t> NetMLP::response() {      
    std::tuple<size_t, neuron_t> neuron {0, (*--layers.end())[0].get_value()};
    for(size_t i{0};i < (*--layers.end()).size();i++) 
        if(std::get<neuron_t>(neuron) < (*--layers.end())[i].get_value()) 
            neuron = {i, (*--layers.end())[i].get_value()};
    return neuron;
}
std::tuple<size_t, neuron_t> NetMLP::feedforward(std::vector<neuron_t>& input_values) {
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