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
struct Generator {
    size_t img_width, img_height;
    std::unique_ptr<sf::RenderWindow> window_inst;
    struct Path {
        std::string main;
        std::string train;
        std::string test;
        Path(std::string _main) : main{_main}, train{_main+"/train"}, test{_main+"/test"} {};
    } path;
    Generator(size_t width, size_t height, std::string folder = "dataset") :  
    img_width{width}, img_height{height}, path(folder) {
        auto mkdir{[](std::string name){return "mkdir " + name;}};
        system(mkdir(path.main).c_str());
        system(mkdir(path.train).c_str());
        system(mkdir(path.test).c_str());
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
    void save_label_to_file(uint8_t label, std::string fpath) {
        std::ofstream fout(fpath);
        fout << ((uint8_t)(1 << label));
        fout.close();
    }
   
    std::vector<std::tuple <std::string, size_t, uint8_t>> shape_classes {
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
        auto shape_ptr = std::find_if(shape_classes.begin(), shape_classes.end(), [name](std::tuple <std::string, size_t, uint8_t> elem){return (std::get<std::string>(elem) == name);});
        
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

};
// Пример использования 
int main() {
    std::srand(std::time(nullptr));
    Generator gen(27,27);
    std::vector<std::string> shapes{"triangle", "circle"};
    for(auto x : shapes)
        gen.make_shape(30, x);
    return 0;
}
