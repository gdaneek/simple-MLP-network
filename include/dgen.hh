/**
 * \file dgen.hh 
 * Figure generator (datasets for training and testing)
*/

#ifndef DGEN_HH
#define DGEN_HH

#include <iostream>
#include <fstream>
#include </usr/include/SFML/Graphics.hpp>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <tuple>
#include <filesystem>

/**
 * Structure, to generate figures, for datasets
 * 
*/
struct Generator {
    size_t img_width;   ///< image width
    size_t img_height;  ///< image height
    std::unique_ptr<sf::RenderWindow> window_inst;  ///< Window that can serve as a target for 2D drawing
    
    std::string line = "line";
    std::string triangle = "triangle";
    std::string rectangle = "rectangle";
    std::string pentagon = "pentagon";
    std::string circle = "circle";

    std::vector<std::tuple<std::string, size_t, uint8_t>> shape_classes {
        {static_cast<std::string>("line"), 2, 0}, 
        {static_cast<std::string>("triangle"), 3, 1}, 
        {static_cast<std::string>("rectangle"), 4, 2}, 
        {static_cast<std::string>("pentagon"), 5, 3}, 
        {static_cast<std::string>("circle"), 30, 4}
    };

    /**
     * Path structure, to navigate to datasets
     * 
    */
    struct Path {
        std::string main;   ///< main part of path
        std::string train;  ///< additional, part of path to training dataset
        std::string test;   ///< additional, part of path to testing dataset

        /**
         * Path generator
         * \param[in] _main main part of path
        */
        Path(std::string _main) : main{_main}, train{_main+"/train"}, test{_main+"/test"} {};
    } path;

    /**
     * Generator for figure generator, dataset directories and labels names
     * \param[in] width image width
     * \param[in] height image heigth
     * \param[in] folder path, to dataset folder 
    */
    Generator(size_t width, size_t height, std::string folder = "dataset") :  img_width{width}, img_height{height}, path(folder) {
        std::filesystem::create_directory(path.main);
        std::filesystem::create_directory(path.train);
        std::filesystem::create_directory(path.test);
        make_lnmap();
        window_inst.reset(new sf::RenderWindow(sf::VideoMode(width, height), ""));
    };

    /**
     * Saves figure image to file
     * \param[in] shape figure image
     * \param[in] fpath path to folder, to save image
    */
    void save_shape_to_file(sf::Shape& shape, std::string fpath);

    /**
     * Saves label to file
     * \param[in] label label of image
     * \param[in] fpath path to folder, to save label
    */
    void save_label_to_file(uint8_t label, std::string fpath);

    /**
     * Matches labels with string names
    */
    void make_lnmap(); // label names map - сопоставляет лейблам строковые названия
   
    /**
     * Function, to generate circle, pentagon or triangle
     * \param[in] count number of generated figures
     * \param[in] name name of figure (name of label)
     * \param[in] is_train is it training dataset
    */
    void make_shape(size_t count, std::string name, bool is_train = true);

    /**
     * Function, to generate rectangle, line
     * \param[in] count number of generated figures
     * \param[in] name name of figure (name of label)
     * \param[in] is_train is it training dataset
    */
    void make_rect(size_t count, std::string name, bool is_train = true);

    /**
     * Function, to generate trapeze
     * \param[in] count number of generated figures
     * \param[in] name name of figure (name of label)
     * \param[in] is_train is it training dataset
    */
    void make_trapeze(size_t count, std::string name, bool is_train = true);

    /**
     * Function, to generate all figures (circle, pentagon, triangle, rectangle, line, trapeze)
     * \param[in] count number of every generated figure
     * \param[in] is_train is it training dataset
    */
    void make_all_figures(size_t count, bool is_train);
};

#endif