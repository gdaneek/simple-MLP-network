#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../include/mlp.hh"
#include <stdexcept>
#include <vector>
TEST_CASE("ALL OK") {
	CHECK(1 == 1);
	
}

TEST_CASE("Test NeuronMLP Constructors") {
	neuronval value{5};
	NeuronMLP n1(value);
	REQUIRE(n1.get_value() == value);

	NeuronMLP n2(n1);
	REQUIRE(n1.get_value() == n2.get_value());

	NeuronMLP n3(std::move(n2));
	REQUIRE(n2.get_value() == n3.get_value());

	NeuronMLP emp;
	CHECK(emp.get_value() == neuronval());
}

TEST_CASE("Test methods changing neuron's value") {
	neuronval value{5}, diff{2};
	NeuronMLP n(value);
	n += diff;
	CHECK(n.get_value() = value+diff);

	n *= diff;
	CHECK(n.get_value() == (value+diff)*diff);

	n = diff;
	CHECK(n.get_value() == diff);

	NeuronMLP other(value+diff);
	n = other;
	CHECK(n.get_value() == other.get_value());
}

TEST_CASE("Test Layer MLP constructors") {
	auto sz = 10;
	LayerMLP l(sz);
	REQUIRE(sz == l.size());
}

TEST_CASE("Test Layer MLP constructors") {
	std::vector<NeuronMLP> nv(5, NeuronMLP(5));
	LayerMLP l(nv);
	REQUIRE(l.size() == nv.size());
	for(size_t i{0};i < nv.size();i++)
		CHECK(l[i].get_value() == nv[i].get_value());
}

TEST_CASE("Test Layer MLP constructors") {
	LayerMLP l(vector_neuronval{1,2,3,4,5});
	REQUIRE(l.size() == 5);
	for(size_t i{0};i < 5;i++)
		CHECK(l[i].get_value() == i+1);
}

TEST_CASE("Test Layer MLP constructors") {
	vector_neuronval vn{1,2,3,4,5};
	LayerMLP l(vn);
	REQUIRE(l.size() == vn.size());
	for(size_t i{0};i < vn.size();i++)
		CHECK(l[i].get_value() == vn[i]);
}

TEST_CASE("Test Layer MLP operators") {
	vector_neuronval vn{1,2,3,4,5};
	LayerMLP l(vn.size());
	l = std::move(vn);
	vector_neuronval vn2{2,3,4,5,6};
	LayerMLP l2(vn2);
	l2 = std::move(l);
	for(size_t i{0};i < l2.size();i++)
		CHECK(l2[i] = l[i]);
}

TEST_CASE("Check for error in assigning layers of different sizes") {
	LayerMLP l1(5), l2(6);
	CHECK_THROWS_AS(l1 = l2,std::runtime_error);
}

// TEST_CASE("Test [] operator") {
// 	NeuronMLP n1(10), n2(5);
// 	std::vector<NeuronMLP> v{n1, n2};
// 	LayerMLP l(v);
// }

TEST_CASE('Test offset applying') {
	std::vector<NeuronMLP> nv{NeuronMLP(5), NeuronMLP(10)};
	LayerMLP l(nv);
	l.apply_offsets();
	for(size_t i{0};i < l.size();i++)
		CHECK(l[i].get_value() != nv[i].get_value());
}