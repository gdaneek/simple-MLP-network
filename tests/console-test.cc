#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../include/console.hh"
#include "StringSimilaryIndex.hpp"
#include <stdexcept>
#include <vector>
#include <filesystem>
#define CHECK_ERROR_MSG(func_call, expected_msg) try {func_call;} catch(std::runtime_error &e){CHECK(std::string(e.what()) == std::string(expected_msg));} 


TEST_CASE("Test console print") {
	std::string msg{"test print method"};
    std::ostringstream oss;
    Console cons(oss);
    cons.print(msg);
    CHECK(msg == oss.str());
}

TEST_CASE("Test console println") {
	std::string msg{"test print method"};
    std::ostringstream oss;
    Console cons(oss);
    cons.println(msg);

    CHECK(msg+"\n" == oss.str());
}

TEST_CASE("Test new net") {
    std::ostringstream oss;
    Console cons(oss);
    cons.new_net({});
    std::string expected{"Empty net is initialized"};

    CHECK(similarity_index(expected, oss.str()) >= 0.5);
}

TEST_CASE("Test make no net") {
    std::ostringstream oss;
    Console cons(oss);
    cons.make({});
    std::string expected{"No net loaded"};

    CHECK(similarity_index(expected, oss.str()) >= 0.5);
}

TEST_CASE("Test make valid") {
    std::ostringstream oss;
    Console cons(oss);
    cons.new_net({});
    cons.make({});
    std::string expected{"Empty net is initialized"};
    CHECK(similarity_index(expected, oss.str()) >= 0.5);
}

TEST_CASE("Test save no net") {
    std::ostringstream oss;
    Console cons(oss);
    cons.save({});
    std::string expected{"No net loaded"};

    CHECK(similarity_index(expected, oss.str()) >= 0.5);
}


TEST_CASE("Test save valid") {
    std::ostringstream oss;
    Console cons(oss);
    cons.new_net({});
    cons.save({"test_model.mlp"});
    std::string expected{"Empty net is initialized\n Model saved to test_model.mlp"};

    CHECK(similarity_index(expected, oss.str()) >= 0.5);
    REQUIRE(std::filesystem::exists("test_model.mlp"));
}

TEST_CASE("Test train no args") {
    std::ostringstream oss;
    Console cons(oss);
    cons.train({});
    std::string expected{"train command requires at least 1 argument"};

    CHECK(similarity_index(expected, oss.str()) >= 0.5);
}

TEST_CASE("Test add no net") {
    std::ostringstream oss;
    Console cons(oss);
    cons.add_layer({});
    std::string expected{"No net loaded"};

    CHECK(similarity_index(expected, oss.str()) >= 0.5);
}

TEST_CASE("Test add no args") {
    std::ostringstream oss;
    Console cons(oss);
    cons.new_net({});
    cons.add_layer({});
    std::string expected{"Empty net is initialized add_layer command requires at least 2 argument"};

    CHECK(similarity_index(expected, oss.str()) >= 0.5);
}

TEST_CASE("Test add invalid activation") {
    std::ostringstream oss;
    Console cons(oss);
    cons.new_net({});
    cons.add_layer({"5", "__undefined__"});
    std::string expected{"Empty net is initialized Failed process command: Requested activation function with name does not exist"};

    CHECK(similarity_index(expected, oss.str()) >= 0.5);
}

TEST_CASE("Test load no args") {
    std::ostringstream oss;
    Console cons(oss);
    cons.load({});
    std::string expected{"load command requires at least 1 argument"};

    CHECK(similarity_index(expected, oss.str()) >= 0.5);
}

TEST_CASE("Test load bad file path") {
    std::ostringstream oss;
    Console cons(oss);
    cons.load({"..."});
    std::string expected{"Model builder: This file is not an MLP model file"};

    CHECK(similarity_index(expected, oss.str()) >= 0.5);
}

TEST_CASE("Test load valid") {
    std::ostringstream oss;
    Console cons(oss);
    cons.new_net({});
    cons.add_layer({"15", "ReLU"});
    cons.add_layer({"5", "sigmoid"});
    cons.add_layer({"30", "softmax"});
    cons.make({});
    cons.save({"test_load.mlp"});
    cons.load({"test_load.mlp"});
    std::string expected{
        R"(Empty net is initialized 
        OK add layer
        OK add layer
        OK add layer
        Model saved to test_load.mlp
        Net has been successfully loaded. 
        Message from file:)"};

    CHECK(similarity_index(expected, oss.str()) >= 0.5);
}

TEST_CASE("Test net info no net") {
    std::ostringstream oss;
    Console cons(oss);
    cons.net_info({});
    std::string expected{"No net loaded"};

    REQUIRE(similarity_index(expected, oss.str()) >= 0.5);
}

TEST_CASE("Test net info valid") {
    std::ostringstream oss;
    Console cons(oss);
    cons.new_net({});
    cons.add_layer({"15", "ReLU"});
    cons.add_layer({"7", "sigmoid"});
    cons.add_layer({"10", "tanh"});
    cons.add_layer({"700", "softmax"});
    cons.make({});
    cons.net_info({});
    std::string answer {
        R"(Empty net is initialized
        Net information: 
        Number of layers: 4 
        Layer 0: 15 neurons with ReLU 
        Layer 1: 7 neurons with sigmoid 
        Layer 2: 10 neurons with tanh 
        Layer 3: 700 neurons with softmax 
        Message from file:)"};

    REQUIRE(similarity_index(answer, oss.str()) >= 0.5);
}

TEST_CASE("Test get command invalid command") {
    std::ostringstream oss;
    Console cons(oss);
    
    CHECK(cons.get_command("__undefined__") == nullptr);
}

TEST_CASE("Test get command valid") {
    std::ostringstream oss;
    Console cons(oss);

    CHECK(cons.get_command("save") != nullptr);
}