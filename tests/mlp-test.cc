#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../include/mlp.hh"
#include <stdexcept>
#include <vector>


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
	CHECK(n.get_value() == value+diff);

	n *= diff;
	CHECK(n.get_value() == (value+diff)*diff);

	n = diff;
	CHECK(n.get_value() == diff);

	NeuronMLP other(value+diff);
	n = other;
	CHECK(n.get_value() == other.get_value());
}

TEST_CASE("Test Neuron MLP is castable") {
	NeuronMLP n(5.7);
	CHECK(static_cast<neuron_t>(n) == 5.7);
}

TEST_CASE("Test Layer MLP is iterable") {
	LayerMLP l(std::vector<neuron_t>{1,3,5,7,9});
	size_t it{1};
	for(auto& x : l) {
		REQUIRE(x.get_value() == it);
		it += 2;
	}
}

TEST_CASE("Test Layer MLP constructors") {
	auto sz = 10;
	LayerMLP l(sz);
	REQUIRE(sz == l.size());
}

TEST_CASE("Test Layer size error") {
	CHECK_THROWS_AS(LayerMLP(0),std::runtime_error);
}

TEST_CASE("Test Layer MLP constructors") {
	std::vector<NeuronMLP> nv(5, NeuronMLP(5));
	LayerMLP l(nv);
	REQUIRE(l.size() == nv.size());
	for(size_t i{0};i < nv.size();i++)
		CHECK(l[i].get_value() == nv[i].get_value());
}

TEST_CASE("Test Layer MLP constructors") {
	LayerMLP l(std::vector<neuron_t>{1,2,3,4,5});
	REQUIRE(l.size() == 5);
	for(size_t i{0};i < 5;i++)
		CHECK(l[i].get_value() == i+1);
}

TEST_CASE("Test Layer MLP constructors") {
	std::vector<neuron_t> vn{1,2,3,4,5};
	LayerMLP l(vn);
	REQUIRE(l.size() == vn.size());
	for(size_t i{0};i < vn.size();i++)
		CHECK(l[i].get_value() == vn[i]);
}

TEST_CASE("Test Layer MLP operators") {
	std::vector<neuron_t> vn{1,2,3,4,5};
	LayerMLP l(vn.size());
	l = std::move(vn);
	std::vector<neuron_t> vn2{2,3,4,5,6};
	LayerMLP l2(vn2);
	l2 = std::move(l);
	for(size_t i{0};i < l2.size();i++)
		CHECK(l2[i].get_value() == l[i].get_value());
}

TEST_CASE("Check for error in assigning layers of different sizes") {
	LayerMLP l1(5), l2(6);
	CHECK_THROWS_AS(l1 = l2, std::runtime_error);
}

TEST_CASE("Test mlp links size") {
	MLPLink mlink(5,7);
	REQUIRE(mlink.size() == 5*7);	
}


TEST_CASE("Test mlp links is iterable") {
	MLPLink mlink(5,7);
	size_t iter{0};
	for(auto& x : mlink)++iter;
	REQUIRE(iter == 5*7);
}

TEST_CASE("Test errors during creation connections for layer size 0") {
	CHECK_THROWS_AS(MLPLink(0,7),std::runtime_error);
}

TEST_CASE("Test net MLP constructor") {
	NetMLP net(10, nullptr, 15, nullptr);
	REQUIRE(net.size() == 0);
	net.make();
	REQUIRE(net.size() == 2);
}

TEST_CASE("Test net MLP add method") {
	NetMLP net;
	for(size_t i{0};i < 3;i++) {
		for(size_t j{0};j <= i;j++) {
			net.add(5, nullptr);
		}
		net.make();
		CHECK(net.size() == i+1);
	}
}

TEST_CASE("Test net MLP add method") {
	NetMLP net;
	for(size_t i{0};i < 3;i++) {
		for(size_t j{0};j <= i;j++) {
			net.add(5, nullptr);
		}
		net.make();
		CHECK(net.size() == i+1);
	}
}