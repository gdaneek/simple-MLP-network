#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <filesystem>
#include <sstream>

#include "StringSimilaryIndex.hpp"
#include "doctest.h"
#include "mlp.hh"
#define CHECK_ERROR_MSG(func_call, expected_msg)               \
  try {                                                        \
    func_call;                                                 \
  } catch (std::runtime_error & e) {                           \
    CHECK(std::string(e.what()) == std::string(expected_msg)); \
  }

TEST_CASE("Test make filename method empty") {
  NetMLP net;
  MLPModelSaver mds;
  CHECK(mds.make_filename(net) == "model.mlp");
}

TEST_CASE("Test make filename method not empty") {
  NetMLP net(15, nullptr, 10, nullptr);
  net.add(7, nullptr);
  net.make();
  MLPModelSaver mds;
  CHECK(mds.make_filename(net) == "model-15-7-10.mlp");
}

TEST_CASE("Test save and upload methods") {
  MLPModelSaver mds;
  NetMLP net1(10, activations::table.get_by_name("ReLU"), 5,
              activations::table.get_by_name("softmax"));
  net1.add(17, activations::table.get_by_name("sigmoid"));
  net1.make();
  mds.save_net_to_file(net1, "test_file.mlp", "testing");

  REQUIRE(std::filesystem::exists("test_file.mlp"));

  auto uploaded = mds.upload_net_from_file("test_file.mlp");
  std::string msg = std::get<std::string>(uploaded);

  CHECK(msg == std::string("testing"));

  auto net2 = std::get<NetMLP>(uploaded);
  std::vector<neuron_t> any_values{1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  auto ff1 = net1.feedforward(any_values);
  auto ff2 = net2.feedforward(any_values);

  REQUIRE(std::get<size_t>(ff1) == std::get<size_t>(ff2));
  REQUIRE(std::get<neuron_t>(ff1) == std::get<neuron_t>(ff2));

  std::filesystem::remove("test_file.mlp");
}

TEST_CASE("Test net info method") {
  MLPModelSaver mds;
  NetMLP net(15, activations::table.get_by_name("ReLU"), 700,
             activations::table.get_by_name("softmax"));
  net.add(7, activations::table.get_by_name("sigmoid"));
  net.add(10, activations::table.get_by_name("tanh"));
  net.make();
  std::string answer{
      R"(Net information: 
        Number of layers: 4 
        Layer 0: 15 neurons with ReLU 
        Layer 1: 7 neurons with sigmoid 
        Layer 2: 10 neurons with tanh 
        Layer 3: 700 neurons with softmax 
        Message from file:)"};
  std::stringstream ss;
  mds.show_model_info(net, "", ss);

  CHECK(similarity_index(answer, ss.str()) >= 0.5);
}

TEST_CASE("Test invalid file signature error") {
  std::ofstream fout("test_sign_method.mlp");
  fout << 12345;
  fout.close();
  MLPModelSaver mds;

  CHECK_THROWS_AS(mds.upload_net_from_file("test_sign_method.mlp"),
                  std::runtime_error);
  CHECK_ERROR_MSG(mds.upload_net_from_file("test_sign_method.mlp"),
                  "Model builder: This file is not an MLP model file");
}

TEST_CASE("Test damaged file error") {
  NetMLP net(15, activations::table.get_by_name("ReLU"), 20,
             activations::table.get_by_name("softmax"));
  net.make();
  MLPModelSaver mds;
  mds.save_net_to_file(net, "test_damaged.mlp");

  REQUIRE(std::filesystem::exists("test_damaged.mlp"));

  std::ifstream fin("test_damaged.mlp", std::ios::binary);
  std::vector<char> buff;
  while (fin.good()) buff.push_back(fin.get());
  fin.close();

  std::ofstream fout("test_damaged.mlp", std::ios::binary);
  size_t damage_koeff{15};
  for (size_t i{0}; i < buff.size() - damage_koeff; i++) fout << buff[i];
  fout.close();

  CHECK_THROWS_AS(mds.upload_net_from_file("test_damaged.mlp"),
                  std::runtime_error);
  CHECK_ERROR_MSG(mds.upload_net_from_file("test_damaged.mlp"),
                  "Model builder: This file was damaged or saved incorrectly");

  std::filesystem::remove("test_damaged.mlp");
  std::filesystem::remove("test_sign_method.mlp");
}