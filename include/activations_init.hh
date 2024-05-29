#ifndef ACTIVATIONS_INIT_HH
#define ACTIVATIONS_INIT_HH
#include "mlp.hh"
#include <vector>
struct __activations_table__ {
    std::vector<activation_ptr> table;
    __activations_table__(std::vector<activation_ptr> _table) : table{_table} {}; 
    ~__activations_table__() {for(size_t i{0};i < table.size();i++)delete table[i];table.clear();}
    std::vector<activation_ptr>::iterator begin(){return table.begin();}
    std::vector<activation_ptr>::iterator end(){return table.end();}
} __activations_table__{{
    new activations::Empty(),
    new activations::ReLU(),
    new activations::Tanh(),
    new activations::Sigmoid(),
    new activations::Softmax()
}};
#endif