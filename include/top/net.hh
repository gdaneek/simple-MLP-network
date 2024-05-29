#ifndef NET_HH
#define NET_HH
#include <vector>
#include <cstdint>
#include <memory>

using neuronval = double;
using vector_neuronval = std::vector<neuronval>;
using weight = double;
using weight_vector = std::vector<weight> ;
using weight_matrix = std::vector<std::vector<weight>>;

/**
 * Polymorphic neuron class <br>
 * Defines the main methods and fields that each derived neuron class must have
*/
class Neuron {          
  protected:                                                                                                                                
    neuronval value; //< current neuron value
  public:       
    /**
     * allows to assign a neuron the new value
     * \param value new value
     * \return this
    */
    virtual Neuron& operator =(const neuronval value) = 0;       
    /**
     *  Requires += operator
     * \param value to be added
     * \return this
    */       
    virtual Neuron& operator +=(const neuronval value) = 0;  
    /**
     *  Requires *= operator
     * \param value to multiply by
     * \return this
    */       
    virtual Neuron& operator *=(const neuronval value) = 0; 
    /**
     * Requires a cast to value
     * \return neuronval value
    */  
    explicit virtual operator neuronval() const = 0; // оператор приведения типа. explicit запрещает неявное использование
    /**
     * Getter for neuron value
     * \return value of a neuron
    */
    const neuronval get_value() const {return value;}
};                   

/**
 * Proxy layer iterator class <bt>
 * Does not implement any methods, 
 * only calls the corresponding methods of the layer iterator derived class object
*/
class LayerIteratorProxy;

/**
 * Layer interface <br>
*/
class Layer {      
  public:                     
    /**
     * Requires a layer size calculation method
     * \return Layer size 
    */                         
    virtual size_t size() const = 0;   
    // virtual Neuron& operator [](const int64_t i) = 0;  // в новой версии не является обязательным

    /**
     * Layer iterator interface
    */
    struct iterator {
        Neuron* ptr;
        virtual iterator& operator++(int) = 0; 
        virtual iterator& operator--(int) = 0; 
        virtual iterator& operator++() = 0; 
        virtual iterator& operator--() = 0; 
        virtual Neuron& operator*() = 0;
        virtual Neuron* operator->() = 0;
        virtual bool operator==(iterator& other) = 0;
        virtual bool operator!=(iterator& other) = 0;
    };              

    /**
     * requires a method that returns an iterator to the beginning of the layer
     * \return iterator pointing to the first neuron
    */
    virtual LayerIteratorProxy begin() = 0;

     /**
     * requires a method that returns an iterator to the ending of the layer
     * \return iterator pointing to the memory area behind the last neuron
    */
    virtual LayerIteratorProxy end() = 0;
};        


class LayerIteratorProxy {  // unique ptr использовать нельзя. Ошибка - used deleted funcs
  std::shared_ptr<Layer::iterator> ptr; //< pointer to an object derived from the layer iterator class
public:
  explicit operator Layer::iterator&() {
    return *(ptr.get());
  }
  LayerIteratorProxy(Layer::iterator *_ptr) : ptr{std::move(_ptr)} {};
  
  bool operator!=(LayerIteratorProxy other) {
    return (*(ptr.get())) != (*(other.ptr.get())); 
  }
  Layer::iterator& operator++() {
    return (*(ptr.get()))++;
  }

  Layer::iterator& operator--() {
    return (*(ptr.get()))--;
  }

  Neuron& operator*() {
    return *(*(ptr.get()));
  }
};

/**
 * Template polymorphic network class
 * 
*/
template <class LayerArray, class ActivationsArray, class LinksArray, class ResponseType>
class Net {  
  protected:
    LayerArray layers; //< Mandatory presence of a field of layers stored by any structure                                                  
    LinksArray links; //< Mandatory presence of a field of links stored by any structure                                            
    ActivationsArray activations; //< Mandatory presence of a field of activations stored by any structure                 
  public:          
    /**
     * Requires a net size calculation method
     * \return net size 
    */          
    virtual size_t size() const = 0;   

    /**
     * Requires a method that implements direct propagation
    */                                                           
    virtual void feedforward() = 0; 

    /**
     * Requires a method that returns a network response
     * \return ResponseType's value
    */
    virtual ResponseType response() = 0;
    
    /**
     * Requires a method that implements weight updating (training)
    */            
    virtual void tng() = 0;  
};

#endif