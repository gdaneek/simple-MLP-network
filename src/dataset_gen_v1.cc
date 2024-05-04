#include <iostream>
#include <fstream>
#include </usr/include/SFML/Graphics.hpp>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <string>
#include <memory>
#include <map>
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
    std::map<size_t, std::string> polygon_names = {
    std::make_pair(3, "triangle"),
    std::make_pair(4, "rectangle"),
    std::make_pair(30, "circle")
    };
    // Генерирующие функции
    void make_polygon(size_t count, size_t vertex, bool is_train = true) {
        for(size_t i{0};i < count;i++) {
            size_t rad = (std::rand()%(img_width/2-6))+5;
            sf::CircleShape polygon(rad, vertex); 
            polygon.move(std::rand()%(img_width-rad*2), std::rand()%(img_width-rad*2));   
            save_shape_to_file(polygon, ((is_train)? path.train : path.test)+"/"+polygon_names[vertex]+"_"+std::to_string(i)+".png");
        }  
    }

};
// Пример использования 

/*
int main() {
    std::srand(std::time(nullptr));
    Generator gen(27,27);
    std::vector<size_t> vertex{3,4, 30};
    for(auto x : vertex)
        gen.make_polygon(30, x);
    return 0;
}
*/