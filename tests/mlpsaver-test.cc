#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../include/mlp.hh"

#include <stdexcept>
#include <vector>
#include <sstream>

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
    NetMLP net1(10, activations::table.get_by_name("ReLU"), 5, activations::table.get_by_name("softmax"));
    net1.add(17, activations::table.get_by_name("sigmoid"));
    net1.make();
    mds.save_net_to_file(net1, "file.mlp", "testing");
    auto uploaded = mds.upload_net_from_file("file.mlp");
    std::string msg = std::get<std::string>(uploaded);

    CHECK((msg == std::string("testing")) == true);
    auto net2 = std::get<NetMLP>(uploaded);
    std::vector<neuron_t> any_values{1,1,1,1,1,1,1,1,1,1};
    auto ff1 = net1.feedforward(any_values);
    auto ff2 = net2.feedforward(any_values);
    REQUIRE((std::get<size_t>(ff1) == std::get<size_t>(ff2)) == true);
    //  std::cout << "ff compare " << std::get<0>(ff1) << " " << std::get<0>(ff2) << std::endl;
    //  std::cout << "ff compare " << std::get<1>(ff1) << " " << std::get<1>(ff2) << std::endl;
}


TEST_CASE("Test net info method") {
  
	MLPModelSaver mds;
    NetMLP net(15, activations::table.get_by_name("ReLU"), 700, activations::table.get_by_name("softmax"));
    net.add(7, activations::table.get_by_name("sigmoid"));
    net.add(10, activations::table.get_by_name("tanh"));
    net.make();
    std::string answer = "\033[95mNet information: \033[39m\n\033[96mNumber of layers: \033[39m4\n\t\033[92mLayer 0: \033[39m15 neurons with ReLU\n\t\033[92mLayer 1: \033[39m7 neurons with sigmoid\n\t\033[92mLayer 2: \033[39m10 neurons with tanh\n\t\033[92mLayer 3: \033[39m700 neurons with softmax\n\033[96mMessage from file:\033[39m\n";
    std::stringstream ss;
    mds.show_model_info(net, "", ss);
    CHECK(ss.str() == answer);
}

TEST_CASE("Test invalid file signature error") {
    std::ofstream fout("test_sign_method.mlp");
    fout << 12345;
    fout.close();
    MLPModelSaver mds;
    CHECK_THROWS_AS(mds.upload_net_from_file("test_sign_method.mlp") ,std::runtime_error);
}

TEST_CASE("Test damaged file error") {
    NetMLP net(15, activations::table.get_by_name("ReLU"), 20, activations::table.get_by_name("softmax"));
    net.make();
    MLPModelSaver mds;
    mds.save_net_to_file(net, "test_damaged.mlp");
    std::ifstream fin("test_damaged.mlp", std::ios::binary);
    std::vector<char> buff;
    while(fin.good())buff.push_back(fin.get());
    fin.close();

    std::ofstream fout("test_damaged.mlp", std::ios::binary);
    size_t damage_koeff{15};
    for(size_t i{0};i < buff.size() - damage_koeff;i++)
        fout << buff[i];
    fout.close();
    CHECK_THROWS_AS(mds.upload_net_from_file("test_damaged.mlp") ,std::runtime_error);


}