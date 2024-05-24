/**
 * \file mlp.hh
 * Multilayer perceptron implementation header file <br>
 *
 */

#ifndef MLP_HH
#define MLP_HH
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <memory>
#include <set>
#include <cstdlib>
#include <ctime>
#include <unordered_set>
#include "top/activations.hh"
#include "top/net_ms.hh"

/**
 * A structure that implements neural connections between two successive layers <br>
 * Structure is quite simplified because it is assumed that it is only necessary to create a certain number of weights for successive layers <br>
 *
 */                                   
class NeuralLink {
  public:                                                   
    weight_matrix weights;  //< implements a weight matrix
    /**
      * Neural connection builder
      * \param input_layer_size first layer size 
      * \param output_layer_size second layer size (must follow the first one!) 
    */
    NeuralLink(size_t input_layer_size, size_t output_layer_size);       
    /**
      * Getting an element by index <br>
      * Supports negative indexing  <br>
      * \param i first layer size 
      * \return vector of input layer neuron weights with index `i`
    */  
    weight_vector& operator [](int64_t i);  
    /**
      * Size of neural connections
      * \return Returns the size of the weight matrix (i.e. the size of the input layer)
    */        
    size_t size();                     
    weight_matrix::iterator begin();                     
    weight_matrix::iterator end();    
    weight_matrix::reverse_iterator rbegin();                     
    weight_matrix::reverse_iterator rend();                           
};                           


/**
 * Structure implementing a neuron <br>
 * Neuron has a value and shift field <br>
 * The value stores the current accumulated amount and is used only in the feedforward algorithm <br>
 * The shift represents the number by which the accumulated amount must be changed. Is a trainable parameter <br>
 *
 */  
class NeuronMLP : public Neuron {
  public:
    neuronval shift; 
    NeuronMLP(neuronval _value);
    NeuronMLP();
    NeuronMLP(NeuronMLP&& other);
    NeuronMLP(NeuronMLP const& other);
    constexpr NeuronMLP& operator=(const NeuronMLP& other);
    Neuron& operator =(neuronval value) override;              
    Neuron& operator +=(neuronval value) override;   
    Neuron& operator *=(neuronval value) override; 
};

/**
 * Structure implementing a layer of neurons <br>
 * Neurons are collected in a dynamic array using vector <br>
 * After creating a layer, the number of neurons cannot be changed <br>
 */                                                                 
class LayerMLP : public Layer {      
  std::vector<NeuronMLP> neurons;                     
  public:                                              
    LayerMLP(size_t size);            
    LayerMLP(std::vector<NeuronMLP>& neurons);  
    LayerMLP(vector_neuronval& neuron_values);  
    LayerMLP(const LayerMLP& other);                                        
    LayerMLP& operator=(vector_neuronval&& values);
    LayerMLP& operator=(LayerMLP&& layer);
    LayerMLP& operator=(const LayerMLP& other);
    void apply_offsets();             
    size_t size() const override;         
    NeuronMLP& operator [](int64_t i) override;  
  
    // struct iterator : public Layer::iterator {
    //   size_t index;
    //   iterator(size_t _index) : index{_index} {};
    //   Layer::iterator& operator++(int) override; 
    //   Layer::iterator& operator--(int) override; 
    //   Layer::iterator& operator++() override; 
    //   Layer::iterator& operator--() override; 
    //   Neuron& operator*() override;
    //   Neuron* operator->() override;
    //   bool operator==(Layer::iterator& other) override;
    //   bool operator!=(Layer::iterator& other) override;  
    // };
    // iterator& begin() override;
    // iterator& end() override;                                  
};                                         

class NetMLP : public Net<std::vector<LayerMLP>, std::vector<activations::fptr>, std::vector<NeuralLink>,  std::tuple<size_t, neuronval>> {      
    friend class MLPModelSaver;      
    size_t layer_indexer; 
    std::set<std::tuple<size_t, size_t, activations::fptr>> table;
    void set_input(weight_vector& values);           
    void calc_output();
    std::tuple<size_t, neuronval> response() override;     
  public:
    NetMLP() = default;
    NetMLP(const NetMLP& other);
    NetMLP(NetMLP&& other);            
    void add(size_t neuron_count, activations::fptr activation, size_t layer_index = 0); 
    NetMLP(size_t in_sz, activations::fptr in, size_t out_sz, activations::fptr out);
    void make();                                                               
    size_t size() const override;                                                              
    void feedforward() override;             
    void tng() override {}; 
                           
    std::tuple<size_t, neuronval> feedforward(vector_neuronval& input_values);           
    void backprop(const std::vector<double>& target, double learning_rate);  
   
};

LayerMLP operator*(LayerMLP& layer, NeuralLink& links);

class MLPModelSaver : public ModelSaver<NetMLP> {
    void check_file_signature(std::ifstream& fin) const;
  public:
    static const std::string make_filename(NetMLP &net);
    std::string save_net_to_file(NetMLP &net, const std::string path="",const std::string msg = "") override;
    NetMLP netmaker(std::vector<std::tuple<size_t, activations::fptr>>& layers, weight_vector& weights, vector_neuronval& shifts) override;
    std::tuple<NetMLP, std::string> upload_net_from_file(std::string path) override;
    void show_model_info(NetMLP& net, std::string msg = "", std::ostream& osteram = std::cout) const;
};


#endif