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
#include "stdarg.h"
typedef double weight;
typedef std::vector<std::vector<weight>> weight_matrix;
typedef std::vector<weight> weight_vector;
                                      
struct NeuralLink {                                                   
   weight_matrix weights;                             
   NeuralLink(size_t input_layer_size, size_t output_layer_size);         
   weight_vector& operator [](int64_t i);        
   size_t size();                     
   weight_matrix::iterator begin();                     
   weight_matrix::iterator end();    
   weight_matrix::reverse_iterator rbegin();                     
   weight_matrix::reverse_iterator rend();                           
};                                                                            
struct Neuron {                                                                                                                                          
    double value;                
    double shift;     
    Neuron(double value = 0);                                                     
    Neuron& operator =(double value);                                             
    void operator +=(double value);    
};                                                                            
struct Layer {                                                                  
    std::vector<Neuron> neurons;                                              
    Layer(size_t size);              
    Layer(std::vector<Neuron>& neurons);    
    Layer(weight_vector& neuron_values);                                      
    weight_vector operator *(weight_vector& weights);               
    //std::vector<double> operator *(NeuralLink& weights);                         
    Layer& operator = (weight_vector values);         
    void apply_offsets();                    
    size_t size();                                                           
    Neuron& operator [](int64_t i);                                           
    std::vector<Neuron>::iterator begin();                                    
    std::vector<Neuron>::iterator end();      
    std::vector<Neuron>::reverse_iterator rbegin();                                       
    std::vector<Neuron>::reverse_iterator rend();                                 
};                                                                             
                                                                               
namespace activations {                                                        
    struct ActivationFunc {                                                    
        virtual void operator ()(Neuron& neuron, ... ) = 0;                  
        virtual void operator ()(Layer& layer, ... ) = 0;                      
    };                                                                         
    struct Empty : public ActivationFunc {                                     
      void operator ()(Neuron& neuron, ... ) override;                         
      void operator ()(Layer& layer, ... ) override;                            
    };                                                                          
    struct ReLU: public ActivationFunc {                                        
      void operator ()(Neuron& neuron, ... ) override;                         
      void operator ()(Layer& layer, ... ) override;                         
    };                                                                        
    struct Tanh: public ActivationFunc {                                      
      void operator ()(Neuron& neuron, ... ) override;                         
      void operator ()(Layer& layer, ... ) override;                           
    };                                                                        
    struct Sigmoid: public ActivationFunc {                                    
      void operator ()(Neuron& neuron, ... ) override;                         
      void operator ()(Layer& layer, ... ) override;                         
    };                                                                         
    struct Softmax : public ActivationFunc {                                   
      double exp_sum{0};                                                       
      void operator ()(Neuron& neuron, ... ) override;                         
      void operator ()(Layer& layer, ... ) override;                           
    };                                                                         
                                                                               
    double derivative(double x0, ActivationFunc& f, double accuracy);                                                       
    double derivative(Neuron &neuron, ActivationFunc& f, double accuracy);     
};                                                                             
                                
struct Net {                                                                    
    size_t layer_indexer;                                                      
    std::set<std::pair<size_t,std::pair<size_t, activations::ActivationFunc*>>> layer_table; 
    std::vector<Layer> layers;                                                  
    std::vector<NeuralLink> NeuralLinks;                                        
    std::vector<activations::ActivationFunc*> activations;                      
    void add_layer(size_t neuron_count, activations::ActivationFunc* activation, size_t layer_index); 
    void add_layer(size_t neuron_count, activations::ActivationFunc* activation); 
    Net(size_t input_size, size_t output_size, activations::ActivationFunc* activation_in, activations::ActivationFunc* activation_out);
    Net(){;};
    void make();                                                               
    size_t size();                                                              
    Layer& operator [](int64_t i);                                              
    void set_input(weight_vector& values);                                
    Layer& calc_output();                                                       
    Layer& calc_output(weight_vector& input_values);                    
    std::tuple<size_t, Neuron> Net::result();      
    std::vector<Layer>::iterator begin();                                       
    std::vector<Layer>::iterator end();
    std::vector<Layer>::reverse_iterator rbegin();                                       
    std::vector<Layer>::reverse_iterator rend();
};
weight_vector operator*(Layer& layer, NeuralLink& links);
#endif