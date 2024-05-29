#ifndef ACTIVATIONS_HH
#define ACTIVATIONS_HH
#include <string>
#include <vector>
#include <cmath>
#include "net.hh"

/**
 * a group of classes and functions related to activations
*/
namespace activations {             
  using fname = std::string; //< Specifies the data type for the function name
  
  /**
   * Activation function interface
  */
  class ActivationFunc {             
    friend double derivative(Neuron& neuron, ActivationFunc& f, double accuracy);     
      /**
       * requires defining the application of a function to a neuron
       * \param neuron neuron to which the function needs to be applied
      */                           
      virtual void operator ()(Neuron& neuron) = 0;  
    public:                
      /**
       * requires defining the application of a function to a layer of neurons
       * \param layer layer to which you want to apply the function
      */
      virtual void operator ()(Layer& layer) = 0;     
      /**
       * requires each function to have a name
       * \return name of the function
      */
      virtual fname name() = 0;                 
  };             
  using fptr = ActivationFunc*; //< alias for a pointer to an activation function
  using flink = ActivationFunc&; //< alias for reference to activation function

  /**
   * A class that implements an empty function <br>
   * Does not change the value of the layer and neuron in any way
  */
  class Empty : public ActivationFunc {                                     
      void operator ()(Neuron& neuron) override;        
    public:                 
      void operator ()(Layer& layer) override;  
      fname name() override;                          
  };                         

  /**
   * ReLU function class <br>
   * applies the standard ReLU function to the layer and neurons
  */
  class ReLU: public ActivationFunc {                                        
      void operator ()(Neuron& neuron) override;        
    public:                 
      void operator ()(Layer& layer) override;  
      fname name() override;                                        
  };                   

  /**
   * hyperbolic tangent function class <br>
   * Calculates the hyperbolic tangent value for each neuron in the layer
  */
  class Tanh: public ActivationFunc {                                        
      void operator ()(Neuron& neuron) override;        
    public:                 
      void operator ()(Layer& layer) override;      
      fname name() override;                          
  };                    

  /**
   * sigmoid function class <br>
   * calculates the sigmoid value for each neuron in the layer
  */
  class Sigmoid: public ActivationFunc {                                        
      void operator ()(Neuron& neuron) override;        
    public:                 
      void operator ()(Layer& layer) override;    
      fname name() override;                          
  };                  

  /**
   * softmax function class <br>
   * applies an activation function to each neuron <br>
   * pre-calculates the exponential sum value for the layer
  */
  class Softmax : public ActivationFunc {                                   
      double exp_sum; //< sum of neuron exponents                        
      void operator ()(Neuron& neuron) override;        
    public:                 
      void operator ()(Layer& layer) override;      
      fname name() override;                        
  };                                

  /**
   * implements an activation function table and several methods above it <br>
   * Each function must exist in at least one instance to be used by the model processor <br>
   * therefore the existence of this table is necessary
  */
  struct Table {
    std::vector<fptr> table; //< table of pointers to objects of each activation function
    /**
     * constructor <br>
     * initializes the pointer table with values
    */
    Table(std::vector<fptr> table_) : table{table_} {};
    ~Table(){
      for(auto& af : table)
        delete af;
    }
    /**
     * function that calculates the identifier of the activation function by name
     * \param name name of activation function
     * \return calculated id
    */
    size_t make_id(fname name);
    /**
     * returns a pointer to the activation function whose identifier matches the one passed
     * \param id activation function id
     * \return pointer to the corresponding function
    */
    fptr get_by_id(size_t id);
    /**
     * returns a pointer to the activation function whose name matches the one passed
     * \param name name of the function
     * \return pointer to the corresponding function
    */
    fptr get_by_name(fname name);
    /**
     * returns the name of the function whose identifier matches the one passed
     * \param id id of activation function
     * \return name of corresponding function
    */
    fname name_by_id(size_t id);
    /**
     * returns the name of the function to which the pointer points
     * \param ptr pointer to activation function
     * \return name of this function
    */
    fname name_by_ptr(fptr ptr);
    /**
     * returns the identifier of the function to which the pointer points
     * \param prt pointer to activation function
     * \return identifier of this fucntion
    */
    size_t id_by_ptr(fptr ptr);
};
extern Table table; //< table of activation functions. Defined in activations_table.cc
  
 
  /**
   * calculates the first derivative for the passed activation function with a certain accuracy
   * \param neuron the point at which it is necessary to calculate the derivative
   * \param f activation function for which the derivative is calculated
   * \param accuracy the accuracy with which you need to find the derivative
   * \return derivative value at a point neuron.get_value()
  */
  double derivative(Neuron& neuron, ActivationFunc& f, double accuracy);     
  
};             

#endif