#ifndef ACTIVATIONS_HH
#define ACTIVATIONS_HH
#include <string>
#include <vector>
#include <cmath>
#include "net.hh"

namespace activations {             
  using fname = std::string; 
  
  class ActivationFunc {             
    friend double derivative(Neuron& neuron, ActivationFunc& f, double accuracy);                                
      virtual void operator ()(Neuron& neuron) = 0;  
    public:                
      virtual void operator ()(Layer& layer) = 0;     
      virtual fname name() = 0;                 
  };             
  using fptr = ActivationFunc*;
  using flink = ActivationFunc&;

  class Empty : public ActivationFunc {                                     
      void operator ()(Neuron& neuron) override;        
    public:                 
      void operator ()(Layer& layer) override;  
      fname name() override;                          
  };                         


  class ReLU: public ActivationFunc {                                        
      void operator ()(Neuron& neuron) override;        
    public:                 
      void operator ()(Layer& layer) override;  
      fname name() override;                                        
  };                   


  class Tanh: public ActivationFunc {                                        
      void operator ()(Neuron& neuron) override;        
    public:                 
      void operator ()(Layer& layer) override;      
      fname name() override;                          
  };                    


  class Sigmoid: public ActivationFunc {                                        
      void operator ()(Neuron& neuron) override;        
    public:                 
      void operator ()(Layer& layer) override;    
      fname name() override;                          
  };                  


  class Softmax : public ActivationFunc {                                   
      double exp_sum;                                                          
      void operator ()(Neuron& neuron) override;        
    public:                 
      void operator ()(Layer& layer) override;      
      fname name() override;                        
  };                                

  struct Table {
    std::vector<fptr> table;
    Table(std::vector<fptr> table_) : table{table_} {};
    ~Table(){for(auto& af : table)delete af;}
    size_t make_id(fname name);
    fptr get_by_id(size_t id);
    fptr get_by_name(fname name);
    fname name_by_id(size_t id);
    fname name_by_ptr(fptr ptr);
    size_t id_by_ptr(fptr ptr);
};
extern Table table;
  
 

  double derivative(Neuron& neuron, flink f, double accuracy);     
};             


//#ifdef ENABLE_ACTIVATIONS_TABLE
 
//#endif


#endif