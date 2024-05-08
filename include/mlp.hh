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

typedef double weight;
typedef std::vector<std::vector<weight>> weight_matrix;
typedef std::vector<weight> weight_vector;

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
class Neuron {                                                                                                                                          
    double value;   //< Accumulated value during feedforward
  public:
    double shift;   //< Accumulated value offset. Trainable
    /**
      * Neuron constructor
      * \param value the value with which to create a neuron
    */
    Neuron(double value = 0);              
    Neuron() = delete;                    
    /**
      * Sets an attribute `value` to a new value
      * \param value new attribute value
      * \return link to the current object
    */                    
    Neuron& operator =(double value);           
     /**
      * Adds any value to the `value` attribute
      * \param value value to be added
      * \return link to the current object
    */                                                     
    Neuron& operator +=(double value);   
    /**
      * returns the current accumulated value
      * \return `value` attribute
    */            
    double get_value();
};                                                                            
class Layer {                                                                  
    std::vector<Neuron> neurons;
  public:                                              
    Layer(size_t size);              
    Layer(std::vector<Neuron>& neurons);    
    Layer(weight_vector& neuron_values);                                      
    weight_vector operator *(weight_vector& weights);               
                             
    Layer& operator = (weight_vector values);
    // тут надо Layer& operator = (weight_vector& values); и Layer& operator = (weight_vector&& values);
    void apply_offsets();                    
    size_t size();                                                           
    Neuron& operator [](int64_t i);                                           
    std::vector<Neuron>::iterator begin();                                    
    std::vector<Neuron>::iterator end();      
    std::vector<Neuron>::reverse_iterator rbegin();                                       
    std::vector<Neuron>::reverse_iterator rend();                                 
};                                                                             
                                                                               
namespace activations {             

  class ActivationFunc {             
      friend double derivative(double x0, ActivationFunc& f, double accuracy);       
      friend double derivative(Neuron &neuron, ActivationFunc& f, double accuracy);                                     
      virtual void operator ()(Neuron& neuron) = 0;  
    public:                
      virtual void operator ()(Layer& layer) = 0;                      
  };                                                                         
  class Empty : public ActivationFunc {                                     
      void operator ()(Neuron& neuron) override;        
    public:                 
      void operator ()(Layer& layer) override;                            
  };                                                                          
  class ReLU: public ActivationFunc {                                        
      void operator ()(Neuron& neuron) override;        
    public:                 
      void operator ()(Layer& layer) override;                                       
  };                                                                        
  class Tanh: public ActivationFunc {                                        
      void operator ()(Neuron& neuron) override;        
    public:                 
      void operator ()(Layer& layer) override;                             
  };                                                                        
  class Sigmoid: public ActivationFunc {                                        
      void operator ()(Neuron& neuron) override;        
    public:                 
      void operator ()(Layer& layer) override;                           
  };                                                                         
  class Softmax : public ActivationFunc {                                   
      double exp_sum;                                                          
      void operator ()(Neuron& neuron) override;        
    public:                 
      void operator ()(Layer& layer) override;                           
  };                                

  double derivative(double x0, ActivationFunc& f, double accuracy);        // должны быть френдами класса активации                                               
  double derivative(Neuron &neuron, ActivationFunc& f, double accuracy);     
};                                                                             
                                
class Net {                                                                    
    size_t layer_indexer; 
    std::set<std::tuple<size_t, size_t, activations::ActivationFunc*>> table;
    void set_input(weight_vector& values);           
    void calc_output();
    std::tuple<size_t, double> result();     
  public:
    std::vector<Layer> layers;                                                  
    std::vector<NeuralLink> NeuralLinks;                                        
    std::vector<activations::ActivationFunc*> activations;                      
    void add(size_t neuron_count, activations::ActivationFunc* activation, size_t layer_index = 0); 
    Net(size_t input_size, size_t output_size, activations::ActivationFunc* activation_in, activations::ActivationFunc* activation_out);
    //Net() = default;
    void make();                                                               
    size_t size();                                                              
    Layer& operator [](int64_t i);                                              
    std::tuple<size_t, double> feedforward(weight_vector& input_values);             
    std::vector<Layer>::iterator begin();                                       
    std::vector<Layer>::iterator end();
    std::vector<Layer>::reverse_iterator rbegin();                                       
    std::vector<Layer>::reverse_iterator rend();
};
weight_vector operator*(Layer& layer, NeuralLink& links);
#endif