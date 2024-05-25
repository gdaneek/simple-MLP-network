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

using label_t = uint8_t;
struct Generator {
    size_t img_width, img_height;
    std::unique_ptr<sf::RenderWindow> window_inst;
    struct Path {
        std::string main;
        std::string train;
        std::string test;
        Path(std::string _main) : main{_main}, train{_main+"/train"}, test{_main+"/test"} {};
    } path;
    Generator(size_t width, size_t height, std::string folder = "dataset") :  img_width{width}, img_height{height}, path(folder) {
        std::filesystem::create_directory(path.main);
        std::filesystem::create_directory(path.train);
        std::filesystem::create_directory(path.test);
        make_lnmap();
        window_inst.reset(new sf::RenderWindow(sf::VideoMode(width, height), ""));
    };
    std::vector<size_t> rvalues(size_t size, size_t mod = -1, size_t min_value = 0) {
        std::vector<size_t> res(size);
        for(size_t i{0};i < size;i++) {
            size_t val = std::rand()%mod;
            while(val < min_value)val = std::rand()%mod;
            res[i] = val;
        }
        return res;
    }
    void save_shape_to_file(sf::Shape& shape, std::string fpath) {
         window_inst->draw(shape);
         window_inst->capture().saveToFile(fpath);
        window_inst->clear();
    }
    void save_label_to_file(label_t label, std::string fpath) {
        std::ofstream fout(fpath);
        fout << ((label_t)(1ULL << label));
        fout.close();
    }
    void make_lnmap() { // label names map - сопоставляет лейблам строковые названия
        std::ofstream fout(path.main+"/setup.lnmap", std::ios::binary);    // СНАЧАЛА НАПИСАТЬ РАЗМЕР ЛЕЙБЛЫ!  + сепаратор
         auto byte_writer{[](auto value, std::ofstream& f) {f.write((char*)&value, sizeof(value));}};
         char sep = '\n';
         uint8_t label_sz = sizeof(label_t);
         byte_writer(label_sz, fout);
         byte_writer(sep, fout);
         for(auto& shape_class : shape_classes) {
            byte_writer(std::get<label_t>(shape_class), fout);
            fout << std::get<std::string>(shape_class);
            byte_writer(sep, fout);
         }

        fout.close();

    }
   
    std::vector<std::tuple <std::string, size_t, label_t>> shape_classes {
      {"line", 2, 0}, 
      {"triangle", 3, 1}, 
    //  {"square", 4, 7},
      {"rhombus", 4, 2},
      {"rectangle", 4, 3},
      {"trapeze", 4, 4},
      {"pentagon", 5, 5},
      {"circle", 30, 6}
    };
    void make_shape(size_t count, std::string name, bool is_train = true) {
        auto shape_ptr = std::find_if(shape_classes.begin(), shape_classes.end(), [name](std::tuple <std::string, size_t, label_t> elem){return (std::get<std::string>(elem) == name);});
        
        // suitable for shapes: triangle, pentagon, circle
        for(size_t i{0};i < count;i++) {

            auto vertex = std::get<size_t>(*shape_ptr);
            auto label = std::get<uint8_t>(*shape_ptr);
            size_t rad = (std::rand()%(img_width/2-6))+5;
            sf::CircleShape polygon(rad, vertex); 
            polygon.move(std::rand()%(img_width-rad*2), std::rand()%(img_width-rad*2));
            save_shape_to_file(polygon, ((is_train)? path.train : path.test)+"/shape_"+std::to_string(i)+"_"+name+".png");
            save_label_to_file(label, ((is_train)? path.train : path.test)+"/label_"+std::to_string(i)+"_"+name+".label");
        }
    }
    void make_rect(size_t count, std::string name, bool is_train = true) {  // suitable for line, rect
         auto shape_ptr = std::find_if(shape_classes.begin(), shape_classes.end(), [name](std::tuple <std::string, size_t, label_t> elem){return (std::get<std::string>(elem) == name);});

        sf::RectangleShape line(sf::Vector2f(150, 1));
        for(size_t i{0};i < count;i++) {

            auto vertex = std::get<size_t>(*shape_ptr);
            auto label = std::get<uint8_t>(*shape_ptr);
            size_t rad = (std::rand()%(img_width/2-6))+5;
            sf::CircleShape polygon(rad, vertex); 
            polygon.move(std::rand()%(img_width-rad*2), std::rand()%(img_width-rad*2));
            save_shape_to_file(polygon, ((is_train)? path.train : path.test)+"/shape_"+std::to_string(i)+"_"+name+".png");
            save_label_to_file(label, ((is_train)? path.train : path.test)+"/label_"+std::to_string(i)+"_"+name+".label");
        }

    }

};
// Пример использования 
int main() {
    std::srand(std::time(nullptr));
    Generator gen(27,27);
    std::vector<std::string> shapes{"triangle", "circle", "pentagon"};
    for(auto x : shapes)
        gen.make_shape(30, x);

    for(auto x : shapes)
        gen.make_shape(5, x, false);
    return 0;
}
