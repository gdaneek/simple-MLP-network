#include "../include/mlp.hh"

NeuralLink::NeuralLink(size_t input_layer_size, size_t output_layer_size) {
    weights = weight_matrix(input_layer_size,weight_vector(output_layer_size));
    for(size_t i{0};i < input_layer_size;i++)
        for(size_t j{0};j < output_layer_size;j++)
            weights[i][j] = std::rand();
}
size_t NeuralLink::size() {return weights.size();}
weight_vector& NeuralLink::operator [](int64_t i) {return weights[(i < 0)? weights.size()-i : i];  }
weight_matrix::iterator NeuralLink::begin() {return weights.begin();}
weight_matrix::iterator NeuralLink::end() {return weights.end();}
Neuron::Neuron(double value) {
        this->shift = std::rand();
        this->value = value;
}
Neuron& Neuron::operator =(double value) {
        this->value = value;
        return *this;
}
void Neuron::operator +=(double value) {this->value += value;}

Layer::Layer(size_t size) {
   // std::cout << "init size:" << neurons.size() << std::endl;
    neurons = std::vector<Neuron>(size, Neuron());
}
Layer::Layer(std::vector<Neuron>& neurons) {this->neurons = neurons;}
Layer::Layer(weight_vector& neuron_values) {
    for(auto value : neuron_values)
        neurons.push_back(Neuron(value));
}                                      

weight_vector Layer::operator *(weight_vector& weight) {
    weight_vector mul(neurons.size());
    for(size_t i{0};i < neurons.size();i++)
        mul[i] = weight[i]*neurons[i].value;
    return mul;
}

Layer& Layer::operator =(weight_vector values) {   
    if(values.size() != neurons.size())
        throw std::runtime_error{"The size of the layer and the size of the value vector do not match"};
    for(size_t i{0}; i < values.size();i++)
        neurons[i] = values[i];
    return *this;  
}

void Layer::apply_offsets() {
    for(size_t i{0};i < neurons.size();i++)
        neurons[i] = neurons[i].value + neurons[i].shift;
}

size_t Layer::size() {return neurons.size();}
Neuron& Layer::operator [](int64_t i) {return neurons[(i < 0)? neurons.size()-i : i];}
std::vector<Neuron>::iterator Layer::begin() {return neurons.begin();}
std::vector<Neuron>::iterator Layer::end() {return neurons.end();}
  

double activations::derivative(double x0, ActivationFunc& f, double accuracy = 1e-6) {
      Neuron wrapper1(x0), wrapper2(x0+accuracy);
      f(wrapper2);     f(wrapper1);
      return (wrapper2.value - wrapper1.value) / accuracy;
}
double activations::derivative(Neuron &neuron, ActivationFunc& f, double accuracy = 1e-6) {
      Neuron wrapper(neuron.value+accuracy);
      f(wrapper);
      return (wrapper.value - neuron.value) / accuracy;
    }
   
void activations::Empty::operator ()(Neuron& neuron, ... ) {;}
void activations::Empty::operator ()(Layer& layer, ... ) {;}

void  activations::ReLU::operator ()(Neuron& neuron, ... ) {
    neuron = (neuron.value < 0)? 0 : neuron.value;
}
void  activations::ReLU::operator ()(Layer& layer, ... )  {
    for(auto n = layer.begin(); n != layer.end();n++)
        (*this)(*n);
}

void activations::Tanh::operator ()(Neuron& neuron, ... )  {
    neuron = std::tanh(neuron.value);
}
void activations::Tanh::operator ()(Layer& layer, ... )  {
   for(auto n = layer.begin(); n != layer.end();n++)
        (*this)(*n);
       
}

void activations::Sigmoid::operator ()(Neuron& neuron, ... )  {
    neuron = 1.0/(1.0+std::exp(-1.0*neuron.value));
}

void activations::Sigmoid::operator ()(Layer& layer, ... )  {
    for(auto n = layer.begin(); n != layer.end();n++)
        (*this)(*n);
}

void  activations::Softmax::operator ()(Neuron& neuron, ... )  {
    va_list args;        
    va_start(args, neuron);   
    double exp_sum{(this->exp_sum)? exp_sum : va_arg(args, double)};
    va_end(args); 
    neuron = 1.0/(1.0+std::exp(-1.0*neuron.value));
    neuron = std::exp(neuron.value)/exp_sum;
}

void activations::Softmax::operator ()(Layer& layer, ... )  {
    this->exp_sum = 0;
     for(auto n = layer.begin(); n != layer.end();n++)
        this->exp_sum+= std::exp(n->value);
    for(auto n = layer.begin(); n != layer.end();n++)
        (*this)(*n);
}
         

void Net::add_layer(size_t neuron_count, activations::ActivationFunc* activation, size_t layer_index) {
    layer_table.insert(std::make_pair(layer_index, std::make_pair(neuron_count, activation)));
}

void Net::add_layer(size_t neuron_count, activations::ActivationFunc* activation) {
    add_layer(neuron_count, activation, ++layer_indexer);
}
        
Net::Net(size_t input_size, size_t output_size, activations::ActivationFunc* activation_in, activations::ActivationFunc* activation_out) {
    layer_indexer = 0;
    add_layer(input_size, activation_in, 0);
    add_layer(output_size, activation_out, -1);
}

void Net::make() {
    for(auto layer : layer_table) {
        layers.push_back(Layer(layer.second.first));
        activations.push_back(layer.second.second);
    }
    for(size_t i{0};i < layers.size()-1;i++) // для последнего layer не делаем линк
        NeuralLinks.push_back(NeuralLink(layers[i].size(),layers[i+1].size()));
    layer_table.clear();
}

size_t Net::size() {return layers.size();}

Layer& Net::operator [](int64_t i) {       return layers[(i < 0)? layers.size()-i : i];}
    
void Net::set_input(weight_vector& values) {layers[0] = values;}

weight_vector operator*(Layer& layer, NeuralLink& links) {
    weight_vector weights(links[0].size(), 0);
    for(size_t neuron_it{0};neuron_it < layer.size();neuron_it++) 
        for(size_t weight_it{0};weight_it < links[neuron_it].size();weight_it++) 
            weights[weight_it] += layer[neuron_it].value * links[neuron_it][weight_it];
        
    return weights;
}

Layer& Net::calc_output() {
    for(size_t i{1};i < layers.size();i++) {
        layers[i] = layers[i-1]*NeuralLinks[i-1];
        layers[i].apply_offsets();
        (*activations[i])(layers[i]);
    }
    return layers[-1];
}

Layer& Net::calc_output(weight_vector& input_values) {
    set_input(input_values);
    return calc_output();
}

std::pair<size_t, Neuron> Net::result() {       
    std::pair<size_t, Neuron> greatest_neuron{std::make_pair(0, layers[-1][0])};
        for(size_t i{0};i < layers[-1].size();i++)
            if(greatest_neuron.second.value < layers[-1][i].value)
                greatest_neuron = std::make_pair(i, layers[-1][i]);
    return greatest_neuron;
}

std::vector<Layer>::iterator Net::begin() {return layers.begin();}
std::vector<Layer>::iterator Net::end() {return layers.end();}