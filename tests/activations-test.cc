#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include<iostream>
#include "../include/top/activations.hh"
#include "../include/mlp.hh"
#include "doctest.h"
#include <stdexcept>
#include <cmath>
TEST_CASE("Test Empty") {
    vector_neuronval values{1,2,3,4,5};
    LayerMLP layer(values);
    activations::Empty act;
    act(layer);
    for(size_t i{0};i < layer.size();i++)
        CHECK(layer[i].get_value() == values[i]);
}

TEST_CASE("Test ReLU") {
    vector_neuronval values{-1,-2,3,0,5};
    LayerMLP layer(values);
    activations::ReLU act;
    act(layer);
    for(size_t i{0};i < layer.size();i++)
        CHECK(layer[i].get_value() == std::max(values[i], 0.0));
}

TEST_CASE("Test Sigmoid") {
    vector_neuronval values{-1,-2,3,0,5};
    LayerMLP layer(values);
    activations::Sigmoid act;
    act(layer);
    for(size_t i{0};i < layer.size();i++)
        CHECK(layer[i].get_value() == 1.0/(1+std::exp(-1.0*values[i])));    // не очень хорошо сравнивать double через ==
}

TEST_CASE("Test Tanh") {
    vector_neuronval values{-1,-2,3,0,5};
    LayerMLP layer(values);
    activations::Tanh act;
    act(layer);
    for(size_t i{0};i < layer.size();i++)
        CHECK(layer[i].get_value() == std::tanh(values[i]));
}

TEST_CASE("Test Softmax") {
    vector_neuronval values{-1,-2,3,0,5};
    LayerMLP layer(values);
    activations::Softmax act;
    act(layer);
    double sum_softmax{0}, sum;
    for(size_t i{0};i < layer.size();i++) {
        sum_softmax += layer[i].get_value();
        sum += std::exp(values[i]);
    }

    REQUIRE(sum_softmax == 1.0);

    for(size_t i{0};i < layer.size();i++)
        CHECK((std::exp(values[i])/sum) == layer[i].get_value());
}

TEST_CASE("Test derivative") {


}

using activations::table;

TEST_CASE("Test table make id func") {
    activations::fname some_name{"name"};
    REQUIRE(table.make_id(some_name) != 0);
}

//CHECK_THROWS_AS(smart_sqrt(-1),std::runtime_error);


TEST_CASE("Undefined activation name exception") {
  CHECK_THROWS_AS(table.get_by_name("__undefined"),std::runtime_error);
}


TEST_CASE("Undefined activation id exception") {
  CHECK_THROWS_AS(table.get_by_id(0),std::runtime_error);
}

TEST_CASE("test activations::Table get functions") {
    activations::ReLU relu;

    CHECK(table.get_by_name(relu.name())->name() == relu.name());
    CHECK(table.id_by_ptr(table.get_by_name(relu.name())) == table.make_id(relu.name()));
    CHECK(table.get_by_id(table.make_id(relu.name()))->name() == relu.name());
}