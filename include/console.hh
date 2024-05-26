#ifndef CONSOLE_HH
#define CONSOLE_HH

#include <iostream>
#define ENABLE_ACTIVATIONS_TABLE
#include "mlp.hh"
#include "top/activations.hh"
#include "learning.hh"
#include "top/net_ms.hh"
#include "stdarg.h"
#include <set>
#include <unordered_set>
#include <initializer_list>
#include <memory>
#include <map>
#include <ctime>
#include <iostream>
#include "../dependencies/SFML/include/Graphics.hpp"
#include <filesystem>       // ВЕРСИЯ CPP не ниже 17
namespace fs = std::filesystem;

std::vector<std::tuple<size_t, std::string>> lnmap_reader(std::string dpath);

class Console {
  MLPModelSaver mds;
  std::unique_ptr<NetMLP> currNet{nullptr};
  static std::map<std::string, void(Console::*)(std::vector<std::string>)> commands;
  bool enable_colors{false};
public:
    void print(std::string msg) ;
    void println(std::string msg);
    void net_info(std::vector<std::string> args);
    void save(std::vector<std::string> args);
    void new_net(std::vector<std::string> args);
    void train(std::vector<std::string> args);
    void add_layer(std::vector<std::string> args);
    void test(std::vector<std::string> args);
    void predict(std::vector<std::string> args);
    void load(std::vector<std::string> args);
    void(Console::*get_command(std::string command))(std::vector<std::string>);

};



#endif