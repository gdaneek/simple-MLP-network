#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "mlp.hh"

#include "doctest.h"

#define CHECK_ERROR_MSG(func_call, expected_msg)               \
  try {                                                        \
    func_call;                                                 \
  } catch (std::runtime_error & e) {                           \
    CHECK(std::string(e.what()) == std::string(expected_msg)); \
  }

TEST_CASE("Test NeuronMLP Constructors") {
  neuron_t value{5};
  NeuronMLP n1(value);
  REQUIRE(n1.get_value() == value);

  NeuronMLP n2(n1);
  REQUIRE(n1.get_value() == n2.get_value());

  NeuronMLP n3(std::move(n2));
  REQUIRE(n2.get_value() == n3.get_value());

  NeuronMLP emp;
  CHECK(emp.get_value() == neuron_t());
}

TEST_CASE("Test methods changing neuron's value") {
  neuron_t value{5}, diff{2};
  NeuronMLP n(value);
  n += diff;
  CHECK(n.get_value() == value + diff);

  n *= diff;
  CHECK(n.get_value() == (value + diff) * diff);

  n = diff;
  CHECK(n.get_value() == diff);

  NeuronMLP other(value + diff);
  n = other;
  CHECK(n.get_value() == other.get_value());
}

TEST_CASE("Test NeuronMLP is castable") {
  NeuronMLP n(5.7);
  CHECK(static_cast<neuron_t>(n) == 5.7);
}

TEST_CASE("Test LayerMLP is iterable") {
  LayerMLP l(std::vector<neuron_t>{1, 3, 5, 7, 9});
  size_t it{1};
  for (auto& n : l) {
    REQUIRE(static_cast<neuron_t>(n) == it);
    it += 2;
  }
}

TEST_CASE("Test LayerMLP constructors") {
  auto sz = 10;
  LayerMLP l(sz);
  REQUIRE(sz == l.size());
}

TEST_CASE("Test LayerMLP size error") {
  CHECK_THROWS_AS(LayerMLP(0), std::runtime_error);
  CHECK_ERROR_MSG(LayerMLP(0), "Layer size cannot be 0");
}

TEST_CASE("Test LayerMLP constructors") {
  std::vector<NeuronMLP> nv(5, NeuronMLP(5));
  LayerMLP l(nv);
  REQUIRE(l.size() == nv.size());
  for (size_t i{0}; i < nv.size(); i++)
    CHECK(l[i].get_value() == nv[i].get_value());
}

TEST_CASE("Test LayerMLP constructors") {
  LayerMLP l(std::vector<neuron_t>{1, 2, 3, 4, 5});
  REQUIRE(l.size() == 5);
  for (size_t i{0}; i < 5; i++) CHECK(l[i].get_value() == i + 1);
}

TEST_CASE("Test LayerMLP constructors") {
  std::vector<neuron_t> vn{1, 2, 3, 4, 5};
  LayerMLP l(vn);
  REQUIRE(l.size() == vn.size());
  for (size_t i{0}; i < vn.size(); i++) CHECK(l[i].get_value() == vn[i]);
}

// TEST_CASE("Check for error in assigning layers of different sizes") {
// 	LayerMLP l1(5), l2(6);

// 	CHECK_THROWS_AS(l1 = l2, std::runtime_error);
// 	CHECK_ERROR_MSG(l1 = l2, "The size of lhs layer and the size of the rhs
// layer do not match");
// }

TEST_CASE("Test mlp links size") {
  MLPLink mlink(5, 7);
  REQUIRE(mlink.size() == 5 * 7);
}

TEST_CASE("Test mlp links is iterable") {
  MLPLink mlink(5, 7);
  size_t iter{0};
  for (auto& x : mlink) ++iter;
  REQUIRE(iter == 5 * 7);
}

TEST_CASE("Test errors during creation connections for layer size 0") {
  CHECK_THROWS_AS(MLPLink(0, 7), std::runtime_error);
  CHECK_ERROR_MSG(MLPLink(0, 7),
                  "trying to create links for a layer with size 0");
}

TEST_CASE("Test netMLP constructor") {
  NetMLP net(10, nullptr, 15, nullptr);
  REQUIRE(net.size() == 0);
  net.make();
  REQUIRE(net.size() == 2);
}

TEST_CASE("Test netMLP add method") {
  NetMLP net;
  for (size_t i{1}; i < 3; i++) {
    for (size_t j{}; j <= i; j++) {
      net.add(5, nullptr);
    }
    net.make();
    CHECK(net.size() == i + 1);
  }
}

TEST_CASE("Test netMLP feedforward and response") {
  MLPModelSaver mds;
  std::vector<std::tuple<size_t, activations::fptr>> layers{
      {4, activations::table.get_by_name("ReLU")},
      {2, activations::table.get_by_name("softmax")}};
  std::vector<weight> weights{0.5, 0.1, 0.2, 0.15, 0.7, 0.85, 0.55, 0.4};
  std::vector<neuron_t> shifts{0.1, 0.5, 0.15, 0.45, 0.9, 0.75};
  auto net = mds.netmaker(layers, weights, shifts);

  std::vector<neuron_t> input{1, 0.5, 0.25, 0.75};

  auto res = net.feedforward(input);

  std::vector<neuron_t> out_layer_values_before_softmax{
      1.1875 + shifts[shifts.size() - 2], 0.6875 + *--shifts.end()};

  REQUIRE(std::get<size_t>(res) == 0);
  // first neuron in out_layer_values_before_softmax is max

  neuron_t exp_sum{};
  for (auto& val : out_layer_values_before_softmax) exp_sum += std::exp(val);

  std::vector<neuron_t> out_layer_values_after_softmax;
  for (size_t i{}; i < out_layer_values_before_softmax.size(); ++i)
    out_layer_values_after_softmax.push_back(
        std::exp(out_layer_values_before_softmax[i]) / exp_sum);

  CHECK(out_layer_values_after_softmax[0] == std::get<neuron_t>(res));
}
