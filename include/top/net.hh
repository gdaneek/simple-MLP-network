#ifndef NET_HH
#define NET_HH

#include <vector>
#include <cstdint>
#include <memory>
#include<concepts>

template<typename T>
concept addable_and_multiplyable = requires(T a, T b) { // поправить на проерку со стандартными конуептами
    { a + b } -> std::convertible_to<T>;
    { a * b } -> std::convertible_to<T>;
    { a - b } -> std::convertible_to<T>;
    { a / b } -> std::convertible_to<T>;
};

template<typename T>
struct check_neuron_value_type {
    static_assert(addable_and_multiplyable<T>, "Type used for neuron value must support addition, multiplication, subtraction, and division.");
    using type = T;
};

using neuron_t = check_neuron_value_type< double >::type;

/**
 * Polymorphic neuron class <br>
 * Defines the main methods and fields that each derived neuron class must have
*/
class Neuron {          
  protected:                                                                                                                                
    neuron_t value; //< current neuron value
  public:       
    /**
     * allows to assign a neuron the new value
     * \param value new value
     * \return this
    */
    virtual Neuron& operator =(const neuron_t value) = 0;       
    /**
     *  Requires += operator
     * \param value to be added
     * \return this
    */       
    virtual Neuron& operator +=(const neuron_t value) = 0;  
    /**
     *  Requires *= operator
     * \param value to multiply by
     * \return this
    */       
    virtual Neuron& operator *=(const neuron_t value) = 0; 
    /**
     * Requires a cast to value
     * \return neuron_t value
    */  
    explicit virtual operator neuron_t() const = 0; // оператор приведения типа. explicit запрещает неявное использование
    /**
     * Getter for neuron value
     * \return value of a neuron
    */
    neuron_t get_value() const {return value;}
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

using weight = double;
using weight_vector = std::vector<weight> ;


class NLinkIteratorProxy;
class NLink {      
  public:                     

    virtual size_t size() const = 0;   
    /**
     * NLink iterator interface
    */
    struct iterator {
        weight* ptr;
        virtual iterator& operator++(int) = 0; 
        virtual iterator& operator--(int) = 0; 
        virtual iterator& operator++() = 0; 
        virtual iterator& operator--() = 0; 
        virtual weight& operator*() = 0;
        virtual weight* operator->() = 0;
        virtual bool operator==(iterator& other) = 0;
        virtual bool operator!=(iterator& other) = 0;
    };              

    /**
     * requires a method that returns an iterator to the beginning of the nlink
     * \return iterator pointing to the element
    */
    virtual NLinkIteratorProxy begin() = 0;

     /**
     * requires a method that returns an iterator to the ending of the nlink
     * \return iterator pointing to the memory area behind the last element
    */
    virtual NLinkIteratorProxy end() = 0;
};        

class NLinkIteratorProxy { 
  std::shared_ptr<NLink::iterator> ptr; 
public:
  explicit operator NLink::iterator&() {
    return *(ptr.get());
  }
  NLinkIteratorProxy(NLink::iterator *_ptr) : ptr{std::move(_ptr)} {};
  
  bool operator!=(NLinkIteratorProxy other) {
    return (*(ptr.get())) != (*(other.ptr.get())); 
  }
  NLink::iterator& operator++() {
    return (*(ptr.get()))++;
  }

  NLink::iterator& operator--() {
    return (*(ptr.get()))--;
  }

  weight& operator*() {
    return *(*(ptr.get()));
  }
};


template<typename T>
concept iterable = requires(T t) {
    { std::begin(t) } -> std::input_iterator;
    { std::end(t) } -> std::input_iterator;
};

template<typename T>
concept layers_cont = iterable<T> && requires(T x) {
    { *std::begin(x) } -> std::convertible_to<const Layer&>;
    { *std::end(x) } -> std::convertible_to<const Layer&>;
};

template<typename T>
concept links_cont = iterable<T> && requires(T x) {
    { *std::begin(x) } -> std::convertible_to<const NLink&>;
    { *std::end(x) } -> std::convertible_to<const NLink&>;
};


template<typename T>
concept callable = requires(T t, Layer& arg) {
    { (*t)(arg) } -> std::same_as<void>;
};

template<typename T>
concept activations_cont = iterable<T> && callable<typename T::value_type>;

template<typename T>
concept not_void = !std::is_void_v<T>;

/**
 * Template polymorphic network class
 * 
*/
template <layers_cont Layers, activations_cont Activations, links_cont NLinks, not_void ResponseType>
class Net {  
  protected:
    Layers layers; //< Mandatory presence of a field of layers stored by any structure                                                  
    NLinks links; //< Mandatory presence of a field of links stored by any structure                                            
    Activations activations; //< Mandatory presence of a field of activations stored by any structure                 
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
    
};

#endif