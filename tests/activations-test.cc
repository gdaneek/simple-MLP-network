#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "top/activations.hh"

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "doctest.h"
#include "mlp.hh"

TEST_CASE("Test Empty") {
  std::vector<neuron_t> values{1, 2, 3, 4, 5};
  LayerMLP layer(values);
  activations::Empty act;
  act(layer);

  for (size_t i{0}; i < layer.size(); i++)
    CHECK(layer[i].get_value() == values[i]);
}

TEST_CASE("Test ReLU") {
  std::vector<neuron_t> values{-1, -2, 3, 0, 5};
  LayerMLP layer(values);
  activations::ReLU act;
  act(layer);
  for (size_t i{0}; i < layer.size(); i++)
    CHECK(layer[i].get_value() == std::max(values[i], 0.0));
}

TEST_CASE("Test Sigmoid") {
  std::vector<neuron_t> values{-1, -2, 3, 0, 5};
  LayerMLP layer(values);
  activations::Sigmoid act;
  act(layer);
  for (size_t i{0}; i < layer.size(); i++)
    CHECK(layer[i].get_value() == 1.0 / (1 + std::exp(-1.0 * values[i])));
}

TEST_CASE("Test Tanh") {
  std::vector<neuron_t> values{-1, -2, 3, 0, 5};
  LayerMLP layer(values);
  activations::Tanh act;
  act(layer);
  for (size_t i{0}; i < layer.size(); i++)
    CHECK(layer[i].get_value() == std::tanh(values[i]));
}

TEST_CASE("Test Softmax") {
  std::vector<neuron_t> values{-1, -2, 3, 0, 5};
  LayerMLP layer(values);
  activations::Softmax act;
  act(layer);
  double sum_softmax{0}, sum;
  for (size_t i{0}; i < layer.size(); i++) {
    sum_softmax += layer[i].get_value();
    sum += std::exp(values[i]);
  }

  REQUIRE(sum_softmax == 1.0);

  for (size_t i{0}; i < layer.size(); i++)
    CHECK((std::exp(values[i]) / sum) == layer[i].get_value());
}

TEST_CASE("Test derivative") {
  auto sigmoid_derivative{[](const neuron_t x) {
    return (1.0 / (1.0 + std::exp(-x))) * (1.0 - (1.0 / (1.0 + std::exp(-x))));
  }};
  activations::Sigmoid sigmoid;
  neuron_t val{5};
  NeuronMLP neuron(val);
  double accuracy{1e-6};

  CHECK(abs(activations::derivative(neuron, sigmoid, accuracy) -
            sigmoid_derivative(val)) <= accuracy);
}

using activations::table;

TEST_CASE("Test table make id func") {
  activations::fname some_name{"name"};
  REQUIRE(table.make_id(some_name) != 0);
}

TEST_CASE("Undefined activation name exception") {
  CHECK_THROWS_AS(table.get_by_name("__undefined"), std::runtime_error);
}

TEST_CASE("Undefined activation id exception") {
  CHECK_THROWS_AS(table.get_by_id(0), std::runtime_error);
}

TEST_CASE("test activations::Table get functions") {
  activations::ReLU relu;

  CHECK(table.get_by_name(relu.name())->name() == relu.name());
  CHECK(table.id_by_ptr(table.get_by_name(relu.name())) ==
        table.make_id(relu.name()));
  CHECK(table.get_by_id(table.make_id(relu.name()))->name() == relu.name());
}