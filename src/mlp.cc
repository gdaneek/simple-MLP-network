#include "../include/mlp.hh"
#include </usr/include/SFML/Graphics.hpp>

NeuralLink::NeuralLink(size_t input_layer_size, size_t output_layer_size) {
    weights = weight_matrix(input_layer_size, weight_vector(output_layer_size));
    for(size_t i{0};i < input_layer_size;i++)
        for(size_t j{0};j < output_layer_size;j++)
            weights[i][j] = (weight)(rand())/RAND_MAX;
}
size_t NeuralLink::size() {return weights.size();}
weight_vector& NeuralLink::operator [](int64_t i) {return weights[(i < 0)? weights.size()+i : i];  }
weight_matrix::iterator NeuralLink::begin() {return weights.begin();}
weight_matrix::iterator NeuralLink::end() {return weights.end();}
weight_matrix::reverse_iterator NeuralLink::rbegin() {return weights.rbegin();}
weight_matrix::reverse_iterator NeuralLink::rend() {return weights.rend();}



Neuron::Neuron(nvalue value) {
        shift = (nvalue)(rand())/RAND_MAX;
        this->value = value;
}
Neuron& Neuron::operator =(nvalue value) {
        this->value = value;
        return *this;
}
Neuron& Neuron::operator +=(nvalue value) {
    this->value += value;
    return *this;
}
nvalue Neuron::get_value() {return value;}


Layer::Layer(size_t size) {neurons = std::vector<Neuron>(size, Neuron(0));}
Layer::Layer(std::vector<Neuron>& neurons) {this->neurons = neurons;}
Layer::Layer(nvalues_vector& neuron_values) {for(auto value : neuron_values)neurons.push_back(Neuron(value));}                                      

Layer& Layer::operator =(nvalues_vector&& values) {   
    if(values.size() != neurons.size())
        throw std::runtime_error{"The size of the layer and the size of the value vector do not match"};
    for(size_t i{0}; i < values.size();neurons[i] = values[i], i++);
    return *this;  
}

void Layer::apply_offsets() {for(Neuron& neuron : neurons)neuron += neuron.shift;}
size_t Layer::size() {return neurons.size();}
Neuron& Layer::operator [](int64_t i) {return neurons[(i < 0)? neurons.size()+i : i];}
std::vector<Neuron>::iterator Layer::begin() {return neurons.begin();}
std::vector<Neuron>::iterator Layer::end() {return neurons.end();}
std::vector<Neuron>::reverse_iterator Layer::rbegin() {return neurons.rbegin();}
std::vector<Neuron>::reverse_iterator Layer::rend() {return neurons.rend();}

double activations::derivative(double x0, ActivationFunc& f, double accuracy = 1e-6) {
    Neuron wrapper1(x0), wrapper2(x0+accuracy);
    f(wrapper2);     f(wrapper1);
    return (wrapper2.get_value() - wrapper1.get_value()) / accuracy;
}
double activations::derivative(Neuron &neuron, ActivationFunc& f, double accuracy = 1e-6) {
    Neuron wrapper(neuron.get_value()+accuracy);
    f(wrapper);
    return (wrapper.get_value() - neuron.get_value()) / accuracy;
}
   
void activations::Empty::operator ()(Neuron& neuron) {}
void activations::Empty::operator ()(Layer& layer) {}
std::string activations::Empty::name(){return "empty";}
void  activations::ReLU::operator ()(Neuron& neuron) { 
    if(neuron.get_value() < 0)neuron = 0;
}
void  activations::ReLU::operator ()(Layer& layer)   { 
    for(Neuron& neuron : layer)(*this)(neuron);
}
std::string activations::ReLU::name(){return "ReLU";}

void activations::Tanh::operator ()(Neuron& neuron){
    neuron = std::tanh(neuron.get_value());
}
void activations::Tanh::operator ()(Layer& layer)  {
    for(Neuron& neuron : layer)(*this)(neuron);
}
std::string activations::Tanh::name(){return "tanh";}

void activations::Sigmoid::operator ()(Neuron& neuron){
    neuron = 1.0/(1.0+std::exp(-1.0*neuron.get_value())); 
}
void activations::Sigmoid::operator ()(Layer& layer)  {
    for(Neuron& neuron : layer)(*this)(neuron);
}
std::string activations::Sigmoid::name(){return "sigmoid";}

void  activations::Softmax::operator ()(Neuron& neuron) {
    neuron = std::exp(neuron.get_value())/exp_sum;
}
void activations::Softmax::operator ()(Layer& layer)    {
     exp_sum = 0;
     for(Neuron& neuron : layer)exp_sum += std::exp(neuron.get_value());
     for(Neuron& neuron : layer)(*this)(neuron);
}
std::string activations::Softmax::name(){return "softmax";}

void Net::add(size_t neuron_count, activation_ptr activation, size_t layer_index) {
   table.insert({((layer_index)? layer_index : ++layer_indexer), neuron_count, activation});
}

Net::Net(size_t input_size, size_t output_size, activation_ptr activation_in, activation_ptr activation_out) {
    layer_indexer = 0;  
    add(input_size, activation_in, ++layer_indexer);
    add(output_size, activation_out, -1);
}

void Net::make() {
    if(!table.size())return;
    size_t prev_layer_size{0};
    std::unordered_set<size_t> indexes;
    for(auto& row : table) {
        if(indexes.count(std::get<0>(row)))
            throw std::runtime_error{"Network creation error: several different layers have the same index"};
        else indexes.insert(std::get<0>(row));
        layers.push_back(std::get<1>(row));
        activations.push_back(std::get<2>(row));
        if(prev_layer_size) 
            links.push_back(NeuralLink(prev_layer_size, (*--layers.end()).size()));
        prev_layer_size = (*--layers.end()).size();
    }
    table.clear();
}

size_t Net::size() {return layers.size();}

Layer& Net::operator [](int64_t i) {return layers[(i < 0)? layers.size()+i : i];}
    

weight_vector operator*(Layer& layer, NeuralLink& links) {
    weight_vector weights(links[0].size(), 0);
    for(size_t neuron_it{0};neuron_it < layer.size();neuron_it++) 
        for(size_t weight_it{0};weight_it < links[neuron_it].size();weight_it++) 
            weights[weight_it] += layer[neuron_it].get_value() * links[neuron_it][weight_it];
    return weights;
}

void Net::set_input(weight_vector& values) {layers[0] = values;}
void Net::calc_output() {
    for(size_t i{1};i < layers.size();i++) {
        layers[i] = layers[i-1]*links[i-1];
        layers[i].apply_offsets();
        (*activations[i])(layers[i]);
    }
}
std::tuple<size_t, nvalue> Net::result() {      
    std::tuple<size_t, nvalue> neuron {0, (*--layers.end())[0].get_value()};
    for(size_t i{0};i < (*--layers.end()).size();i++) 
        if(std::get<nvalue>(neuron) < (*--layers.end())[i].get_value()) 
            neuron = {i, (*--layers.end())[i].get_value()};
    return neuron;
}
std::tuple<size_t, nvalue> Net::feedforward(weight_vector& input_values) {
    set_input(input_values);
    calc_output();
    return result();
}


std::vector<Layer>::iterator Net::begin() {return layers.begin();}
std::vector<Layer>::iterator Net::end() {return layers.end();}
std::vector<Layer>::reverse_iterator Net::rbegin() {return layers.rbegin();}
std::vector<Layer>::reverse_iterator Net::rend() {return layers.rend();}





// ---------------------------------------------------------

/*
std::vector<double> pixels(std::string image_path) {
    std::vector<double> res;
    sf::Image img;
    img.loadFromFile(image_path);
    auto size = img.getSize();
    for(size_t i{0};i < size.x;i++) 
        for(size_t j{0};j < size.y;j++) 
            res.push_back(img.getPixel(i, j).r/255);
    return res;
}  
int main() {
    activations::Empty empty;
    activations::ReLU relu;
    activations::Softmax softmax;
    auto pxs = pixels("triangle0.png");
  //  Net net(pxs.size(), 3, &empty, &softmax);
  //  net.add(7, &relu);
  //  net.make();
  //  net.calc_output(pxs);
  //  for(auto neuron : net[-1])std::cout << neuron.get_value() << std::endl;
    return 0;
}
*/
