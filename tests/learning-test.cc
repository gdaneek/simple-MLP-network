#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../include/dataset.hh"
#include "../include/learning.hh"
#include <stdexcept>
#include <vector>

std::size_t number_of_files_in_directory(std::filesystem::path path) {
    using std::filesystem::directory_iterator;
    return std::distance(directory_iterator(path), directory_iterator{});
}

TEST_CASE("Test vectorize_image") {
    size_t img_width{10}, img_height{15};
    Generator gen = Generator(img_width, img_height);

    sf::RectangleShape line(sf::Vector2f(1, 5));
    gen.save_shape_to_file(line, "dataset_testing/testing_image.png");

    std::vector<neuron_t> image = vectorize_image("./dataset_testing/testing_image.png");

    std::vector<neuron_t> answer{};
    for (size_t i{0}; i < img_width * img_height; i++) {
        if (i <= 4) {
            answer.push_back(1);
        } else {
            answer.push_back(0);
        }
    }
    REQUIRE(image == answer);
}

TEST_CASE("Test vectorize_lable") {
    int lable = 2;
    auto lable_v = vectorize_lable<int>(lable);
    std::vector<neuron_t> answer = {0, 0, 1, 0, 0};

    REQUIRE(lable_v == answer);
}

TEST_CASE("Test circle") {
    size_t img_width{27}, img_height{30};
    std::string folder = "dataset_testing";
    Generator gen(img_width, img_height, folder);

    size_t count_figures = 3;

    gen.make_shape(count_figures, "circle", true);

    REQUIRE(std::filesystem::exists("dataset_testing"));
    REQUIRE(std::filesystem::exists("dataset_testing/test"));
    REQUIRE(number_of_files_in_directory("dataset_testing/test") == 0);
    REQUIRE(number_of_files_in_directory("dataset_testing/train") == count_figures * 2);
}

TEST_CASE("Test backprop") {
    std::vector<double> target{0, 0, 0, 0, 0};
    NetMLP net(27 * 27, activations::table.get_by_name("ReLU"), 5, activations::table.get_by_name("softmax"));
    CHECK_THROWS_AS(net.backprop(target, 0.5), std::runtime_error);
}

// make extra tests with errors