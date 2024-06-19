#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include "../include/dgen.hh"
#include <stdexcept>
#include <vector>

std::size_t number_of_files_in_directory(std::filesystem::path path) {
    using std::filesystem::directory_iterator;
    return std::distance(directory_iterator(path), directory_iterator{});
}

TEST_CASE("Test Generator generator") {
    size_t img_width{10}, img_height{15};
    Generator gen = Generator(img_width, img_height);

    REQUIRE(gen.img_height == img_height);
    REQUIRE(gen.img_height == img_height);
    REQUIRE(gen.path.main == "dataset");
    REQUIRE(gen.path.test == "dataset/test");
    REQUIRE(gen.path.train == "dataset/train");

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


// make extra tests with errors