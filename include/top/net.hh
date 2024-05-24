#ifndef NET_HH
#define NET_HH
#include <vector>
#include <cstdint>
using neuronval = double;
using vector_neuronval = std::vector<neuronval>;
using weight = double;
using weight_vector = std::vector<weight> ;
using weight_matrix = std::vector<std::vector<weight>>;
class Neuron {          
  protected:                                                                                                                                
    neuronval value;   
  public:
    //Neuron(neuronval value = 0);                           
    virtual Neuron& operator =(neuronval value) = 0;              
    virtual Neuron& operator +=(neuronval value) = 0;   
    virtual Neuron& operator *=(neuronval value) = 0;   
    neuronval get_value() {return value;}
};                   


class Layer {      
  public:                                              
    virtual size_t size() const = 0;   
    virtual Neuron& operator [](const int64_t i) = 0;  
    //virtual typename std::vector<NeuronImpl>::iterator begin() = 0;      
    //virtual typename std::vector<NeuronImpl>::iterator end() = 0;      
    // struct iterator {
    //     virtual iterator& operator++(int) = 0; 
    //     virtual iterator& operator--(int) = 0; 
    //     virtual iterator& operator++() = 0; 
    //     virtual iterator& operator--() = 0; 
    //     virtual Neuron& operator*() = 0;
    //     virtual Neuron* operator->() = 0;
    //     virtual bool operator==(iterator& other) = 0;
    //     virtual bool operator!=(iterator& other) = 0;
    // };              
                     
};          

template <class LayerArray, class ActivationsArray, class LinksArray, class ResponseType>
class Net {  
  protected:
    LayerArray layers;                                                  
    LinksArray links;                                        
    ActivationsArray activations;             
  public:          
    virtual size_t size() const = 0;                                                              
    virtual void feedforward() = 0; 
    virtual ResponseType response() = 0;
    virtual void tng() = 0;  
};

#endif