#include <Graphics.hpp>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

using label_t = uint8_t;
// using label_t = char;
// using label_t = int;

struct Generator {
  size_t img_width, img_height;
  std::unique_ptr<sf::RenderWindow> window_inst;
  // std::unique_ptr<sf::Texture> window_inst;
  struct Path {
    std::string main;
    std::string train;
    std::string test;
    Path(std::string _main)
        : main{_main}, train{_main + "/train"}, test{_main + "/test"} {};
  } path;
  Generator(size_t width, size_t height, std::string folder = "dataset")
      : img_width{width}, img_height{height}, path(folder) {
    std::filesystem::create_directory(path.main);
    std::filesystem::create_directory(path.train);
    std::filesystem::create_directory(path.test);
    make_lnmap();
    window_inst.reset(new sf::RenderWindow(sf::VideoMode(width, height), ""));
  };
  std::vector<size_t> rvalues(size_t size, size_t mod = -1,
                              size_t min_value = 0) {
    std::vector<size_t> res(size);
    for (size_t i{0}; i < size; i++) {
      size_t val = std::rand() % mod;
      while (val < min_value) val = std::rand() % mod;
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
    // fout << ((label_t)(1ULL << label));
    fout << (int)label;
    fout.close();
  }
  void
  make_lnmap() {  // label names map - сопоставляет лейблам строковые названия
    std::ofstream fout(
        path.main + "/setup.lnmap",
        std::ios::binary);  // СНАЧАЛА НАПИСАТЬ РАЗМЕР ЛЕЙБЛЫ!  + сепаратор
    auto byte_writer{[](auto value, std::ofstream& f) {
      f.write((char*)&value, sizeof(value));
    }};
    char sep = '\n';
    uint8_t label_sz = sizeof(label_t);
    byte_writer(label_sz, fout);
    byte_writer(sep, fout);
    for (auto& shape_class : shape_classes) {
      byte_writer(std::get<label_t>(shape_class), fout);
      fout << std::get<std::string>(shape_class);
      byte_writer(sep, fout);
    }

    fout.close();
  }

  std::vector<std::tuple<std::string, size_t, label_t>> shape_classes{
      {"line", 2, 0},
      {"triangle", 3, 1},
      //  {"square", 4, 7},
      {"rhombus", 4, 2},
      {"rectangle", 4, 3},
      {"trapeze", 4, 4},
      {"pentagon", 5, 5},
      {"circle", 30, 6}};
  void make_shape(size_t count, std::string name, bool is_train = true) {
    auto shape_ptr =
        std::find_if(shape_classes.begin(), shape_classes.end(),
                     [name](std::tuple<std::string, size_t, label_t> elem) {
                       return (std::get<std::string>(elem) == name);
                     });

    // suitable for shapes: triangle, pentagon, circle
    for (size_t i{0}; i < count; i++) {
      auto vertex = std::get<size_t>(*shape_ptr);
      auto label = std::get<label_t>(*shape_ptr);
      size_t rad = (std::rand() % (img_width / 2 - 6)) + 5;
      sf::CircleShape polygon(rad, vertex);
      polygon.move(std::rand() % (img_width - rad * 2),
                   std::rand() % (img_width - rad * 2));
      save_shape_to_file(polygon, ((is_train) ? path.train : path.test) +
                                      "/shape_" + std::to_string(i) + "_" +
                                      name + ".png");
      save_label_to_file(label, ((is_train) ? path.train : path.test) +
                                    "/label_" + std::to_string(i) + "_" + name +
                                    ".label");
    }
  }
  void make_rect(size_t count, std::string name,
                 bool is_train = true) {  // suitable for line, rect, rhombus
    auto shape_ptr =
        std::find_if(shape_classes.begin(), shape_classes.end(),
                     [name](std::tuple<std::string, size_t, label_t> elem) {
                       return (std::get<std::string>(elem) == name);
                     });

    for (size_t i{0}; i < count; i++) {
      auto vertex = std::get<size_t>(*shape_ptr);
      auto lable = std::get<uint8_t>(*shape_ptr);
      if (lable == 0) {
        int len = std::rand() % img_width + 4;
        sf::RectangleShape line(sf::Vector2f(len, 1));
        line.rotate(std::rand() % 360);
        line.move(std::rand() % (img_width - len / 2),
                  std::rand() % (img_width - len / 2));
        save_shape_to_file(line, ((is_train) ? path.train : path.test) +
                                     "/shape_" + std::to_string(i) + "_" +
                                     name + ".png");
        save_label_to_file(lable, ((is_train) ? path.train : path.test) +
                                      "/label_" + std::to_string(i) + "_" +
                                      name + ".label");
      } else if (lable == 2) {
        int len = std::rand() % (img_width - 3) + 2;
        sf::RectangleShape rhombus(sf::Vector2f(len, len));
        rhombus.move(std::rand() % (img_height - len),
                     std::rand() % (img_width - len));
        // rhombus.rotate(std::rand()%360);
        save_shape_to_file(rhombus, ((is_train) ? path.train : path.test) +
                                        "/shape_" + std::to_string(i) + "_" +
                                        name + ".png");
        save_label_to_file(lable, ((is_train) ? path.train : path.test) +
                                      "/label_" + std::to_string(i) + "_" +
                                      name + ".label");
      } else if (lable == 3) {
        int height = std::rand() % (img_height - 3) + 2;
        int width = std::rand() % (img_width - 3) + 2;
        if (height == width) {
          height += 1;
        }
        sf::RectangleShape rect(sf::Vector2f(height, width));
        rect.move(std::rand() % (img_height - height),
                  std::rand() % (img_width - width));
        // rect.move(std::rand()%(img_height), std::rand()%(img_width));
        // rect.rotate(std::rand()%360);
        save_shape_to_file(rect, ((is_train) ? path.train : path.test) +
                                     "/shape_" + std::to_string(i) + "_" +
                                     name + ".png");
        save_label_to_file(lable, ((is_train) ? path.train : path.test) +
                                      "/label_" + std::to_string(i) + "_" +
                                      name + ".label");
      }
    }
  }

  void make_trapeze(size_t count, std::string name, bool is_train = true) {
    auto shape_ptr =
        std::find_if(shape_classes.begin(), shape_classes.end(),
                     [name](std::tuple<std::string, size_t, label_t> elem) {
                       return (std::get<std::string>(elem) == name);
                     });

    auto vertex = std::get<size_t>(*shape_ptr);
    auto lable = std::get<uint8_t>(*shape_ptr);

    for (size_t i{0}; i < count; i++) {
      sf::ConvexShape trapeze{};
      int height = std::rand() % (img_height - 3) + 2;
      int a = std::rand() % (img_width - 3) + 2;
      int b = std::rand() % (img_width - 3) + 2;

      trapeze.setPointCount(4);

      // define the points
      // trapeze.setPoint(0, sf::Vector2f(0, 0));
      // trapeze.setPoint(1, sf::Vector2f(a, 0));
      // trapeze.setPoint(2, sf::Vector2f(b, height));
      // trapeze.setPoint(3, sf::Vector2f(0, height));

      int start = std::rand() % (img_width / 2);
      int startb = std::rand() % (img_width - b) + b;
      trapeze.setPoint(0, sf::Vector2f(start, 0));
      trapeze.setPoint(1, sf::Vector2f(start + a, 0));
      trapeze.setPoint(2, sf::Vector2f(startb, height));
      trapeze.setPoint(3, sf::Vector2f(startb - b, height));

      // trapeze.move(std::rand()%(img_height - std::max(a, b)),
      // std::rand()%(img_width - height));

      save_shape_to_file(trapeze, ((is_train) ? path.train : path.test) +
                                      "/shape_" + std::to_string(i) + "_" +
                                      name + ".png");
      save_label_to_file(lable, ((is_train) ? path.train : path.test) +
                                    "/label_" + std::to_string(i) + "_" + name +
                                    ".label");
    }
  }

  void make_all_figures(size_t count, bool is_train) {
    std::vector<std::string> shapes{"triangle", "circle", "pentagon"};
    for (auto x : shapes) {
      make_shape(count, x, is_train);
    }
    shapes = {"line", "rectangle", "rhombus"};
    for (auto x : shapes) {
      make_rect(count, x, is_train);
    }
    make_trapeze(count, "trapeze", is_train);
  }
};
// Пример использования
int main() {
  std::srand(std::time(nullptr));
  Generator gen(27, 27);
  gen.make_all_figures(10, false);  // for test
  gen.make_all_figures(10, true);   // for train
}
