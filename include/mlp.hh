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
 * Structure implementing a MLP neuron <br>
 * Adds a shift field containing the amount of shift of the accumulated value before activation <br>
 * Shift is a trainable parameter
 */  
class NeuronMLP : public Neuron {
  public:
    neuronval shift; //< shift value
    /**
     * Constructor <br>
     * \param _value the initial value that will be assigned to the neuron
    */
    NeuronMLP(const neuronval _value);
    /**
     * Constructor <br>
     * implements setting default field values
    */
    NeuronMLP();
    /**
     * Constructor <br>
     * implements the movement of another rvalue neuron
     * \param other rvalue neuron
    */
    NeuronMLP(NeuronMLP&& other);
    /**
     * Constructor <br>
     * implements the copy of another rvalue neuron
     * \param other lvalue neuron
    */
    NeuronMLP(NeuronMLP const& other);
    /**
     * Assignment operator <br>
     * Copies the shift and value fields of another neuron <br>
     * \param other lvalue neuron
    */
    constexpr NeuronMLP& operator=(const NeuronMLP&  other);
    Neuron& operator =(const neuronval value) override;              
    Neuron& operator +=(const neuronval value) override;   // не могут быть virtual и constexpr одновременно
    Neuron& operator *=(const neuronval value) override; 
    explicit operator neuronval() const override; // запрещаю неявное преобразование нейрона к число
};

/**
 * Structure implementing a MLP layer of neurons <br>
 * Neurons are collected in a dynamic array using vector <br>
 * After creating a layer, the number of neurons cannot be changed <br>
*/                                                                 
class LayerMLP : public Layer {      
  std::vector<NeuronMLP> neurons; //< array of MLP neurons                   
public:    
  /**
    * Constructor <br>
    * implements the creation of a vector of neurons with a specified size
    * \param size count of neurons
  */                                          
  LayerMLP(size_t size); 
  /**
    * Constructor <br>
    * Implements the creation of a layer from a vector of neurons
    * \param neurons neurons that need to be added
  */                     
  LayerMLP(std::vector<NeuronMLP>& neurons);  
  /**
   * Constructor <br>
   * Implements the creation of a layer from a lvalue vector of neuron values
   * \param neuron_values vector of neuron's values
  */
  LayerMLP(vector_neuronval& neuron_values);  
  /**
   * Constructor <br>
   * Implements the creation of a layer from a rvalue vector of neuron values
   * \param neuron_values vector of neuron's values
  */
  LayerMLP(vector_neuronval&& neuron_values);  
  /**
   * Constructor <br>
   * Implements the creation of a layer from another layer <br>
   * \param other MLP layer
   * 
  */
  LayerMLP(const LayerMLP& other);       
  /**
   * Implements changing the values ​​of layer neurons <br>
   * The size of the layer and the size of the vector of values ​​must match <br>
   * \param values neuron's values
  */                                 
  LayerMLP& operator=(vector_neuronval&& values);
  /**
   * Implements changing the values ​​of layer neurons <br>
   * The size of the layer and the size of the rvalue layer ​​must match <br>
   * \param layer rvalue MLP layer
  */                            
  LayerMLP& operator=(LayerMLP&& layer);
  /**
   * Implements changing the values ​​of layer neurons <br>
   * The size of the layer and the size of the lvalue layer ​​must match <br>
   * \param other lvalue MLP layer
  */         
  LayerMLP& operator=(const LayerMLP& other);
  /**
   * Shifts the values ​​of all neurons by the amount shift before activation
  */
  void apply_offsets();             
  size_t size() const override;    
  /**
   * implements layer indexing <br>
   * supports negative indexing
   * \param i neuron index
   * \return Neuron from the vector neurons with the corresponding index
   */     
  NeuronMLP& operator [](int64_t i)  /* override */;  
  
  /**
   * MLP Layer iterator
  */
  struct iterator : public Layer::iterator {
    explicit iterator(Neuron* _ptr)  {ptr = _ptr;};
    Layer::iterator& operator++(int) override; 
    Layer::iterator& operator--(int) override; 
    Layer::iterator& operator++() override; 
    Layer::iterator& operator--() override; 
    Neuron& operator*() override;
    Neuron* operator->() override;
    bool operator==(Layer::iterator& other) override;
    bool operator!=(Layer::iterator& other) override;  
  };

  LayerIteratorProxy begin() override;
  LayerIteratorProxy end() override;                                  
};                                         

/**
 * MLP network class
*/
class NetMLP : public Net<std::vector<LayerMLP>, std::vector<activations::fptr>, std::vector<NeuralLink>,  std::tuple<size_t, neuronval>> {      
    friend class MLPModelSaver; //< must be friend to access protected fields
    size_t layer_indexer;  //< indexer for determining the location of a layer in the network
    bool enable_automake{true}; //< allows auto-assembly if the table is not empty
    std::set<std::tuple<size_t, size_t, activations::fptr>> table; //< layer table. Used to build a network
    /**
     * Sets the values ​​on the input layer 
     * \param values values ​​to be set
    */
    void set_input(weight_vector& values);    
    /**
     * calculates the values ​​of output neurons
    */       
    void calc_output();
    /**
     * Returns the network response
     * \return Tuple of numbers: index of the neuron with the largest value and it's value
    */
    std::tuple<size_t, neuronval> response() override;     
  public:
    NetMLP() = default;
    /**
     * allows you to assemble a neural network from another
     * \param other other lvalue MLP net
    */
    NetMLP(const NetMLP& other);
    /**
     * allows you to assemble a neural network from another
     * \param other other rvalue MLP net
    */
    NetMLP(NetMLP&& other); 
    /**
     * Fills out the layer table for subsequent assembly
     * \param neuron_count number of neurons in the added layer
     * \param activation pointer to the activation function to be applied
     * \param layer_index indicates where the layer should be inserted
    */           
    void add(size_t neuron_count, activations::fptr activation, size_t layer_index = 0); 
    /**
     * Constructor <br>
     * Creates a network with at least two layers
    */
    NetMLP(size_t in_sz, activations::fptr in, size_t out_sz, activations::fptr out);
    /**
     * Assembles a network using a layer table
    */
    void make();                 
    /**
     * calculates network size
     * \return layers count
    */                                              
    size_t size() const override;                                                              
    void feedforward() override;             
    void tng() override {}; 
    /**
     * Disables auto-assembly of the network <br>
     * Before using the network, you need to call make so that it is built from the table
    */
    void disable_automake();
    /**
     * Implements forward propagation and returns the network response
     * \param input_values values ​​that need to be set on the input layer
     * \return result of response() method call
    */
    std::tuple<size_t, neuronval> feedforward(vector_neuronval& input_values);          
    /**
     * implements backpropagation algorithm
    */ 
    void backprop(const std::vector<double>& target, double learning_rate);  
   
};

/**
 * A function that implements multiplication of a layer by a mitrice of weights
 * \param layer layer to be multiplied
 * \param links weight matrix
 * \return new layer which is the result of multiplication
*/
LayerMLP operator*(LayerMLP& layer, NeuralLink& links);

/**
 * MLP model processor class <br>
 * Implements saving mlp to a file, <br>
 * reading from a file <br> and assembling networks from several vectors
*/
class MLPModelSaver : public ModelSaver<NetMLP> {
    /**
     * Checks if the file is in .mlp format
     * \param fin file input stream
    */
    void check_file_signature(std::ifstream& fin) const;
  public:
    /**
     * Collects the filename for the network if one was not specified <br>
     * the number of neurons on each layer is used to assemble the name
    */
    static const std::string make_filename(NetMLP &net);
    /**
     * Saves the network to a file
     * \param net network that needs to be saved
     * \param path file name where the network is saved
     * \param msg message of any content
     * \return path where the network was saved if it was not specified
    */
    std::string save_net_to_file(NetMLP &net, const std::string path="",const std::string msg = "") override;
    /**
     * assembles a network of input vectors
     * \param layers number of neurons in a layer and activation to them
     * \param weights vector of weights with which to fill neural connections
     * \param shifts shifts for each neuron
     * \return created mlp network
    */
    NetMLP netmaker(std::vector<std::tuple<size_t, activations::fptr>>& layers, weight_vector& weights, vector_neuronval& shifts) override;
    /**
     * loads network from file
     * \param path path to file with network
     * \return network and accompanying message
    */
    std::tuple<NetMLP, std::string> upload_net_from_file(std::string path) override;
    /**
     * prints network information
     * \param net network information about which needs to be displayed
     * \param msg accompanying message
     * \param osteram output stream
    */
    void show_model_info(NetMLP& net, std::string msg = "", std::ostream& osteram = std::cout) const;
};


#endif