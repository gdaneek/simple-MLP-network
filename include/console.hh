#ifndef CONSOLE_HH
#define CONSOLE_HH

#include "mlp.hh"
#include "top/activations.hh"
#include "learning.hh"
#include "top/net_ms.hh"
#include "../dependencies/SFML/include/Graphics.hpp"

#include <memory>
#include <map>
#include <ctime>
#include <iostream>
#include <fstream>
#include <filesystem>
namespace fs = std::filesystem;

/**
 * Function of obtaining the names of each tag during classification <br>
 * \param dpath path to the .lnmap file with markup
 * \return vector of pairs of labels and their string names
*/
std::vector<std::tuple<size_t, std::string>> lnmap_reader(std::string dpath);

/**
 * The class implements a console for user interaction with mlp networks structs
*/
class Console {
  MLPModelSaver mds; //< model processing class object
  std::unique_ptr<NetMLP> currNet{nullptr}; //< a unique pointer to the current network being processed
  std::string loaded_msg{};
  std::ostream& m_ostream;
  static std::map<std::string, void(Console::*)(std::vector<std::string>)> commands; //< list of defined commands and pointers to the methods that implement them
  bool enable_colors{false}; //< turns on color console
public:
  Console(std::ostream& ostream) : m_ostream{ostream} {};
  /**
   * prints a message to the console
   * \param msg message that needs to be printed to the console
  */
  void print(std::string msg);
  /**
   * prints a message to the console with a line feed
   * \param msg message that needs to be printed to the console
  */
  void println(std::string msg);                          // нужна команда make
  /**
   * prints information about the current network <br>
   * calls the corresponding model handler method <br>
   * requires no arguments
   * \param args arguments passed to the command
  */
  void net_info(std::vector<std::string> args);           // help
  /**
   * Saves the network to a file <br>
   * calls the corresponding model handler method <br>
   * optionally accepts the name of the output file and a message to write to the file
   * \param args arguments passed to the command
  */
  void save(std::vector<std::string> args);
  /**
   * Moves the pointer to a new network with deleting the old one <br>
   * requires no arguments
   * \param args arguments passed to the command
  */
  void new_net(std::vector<std::string> args);
  /**
   * Calls the network training function with the passed parameters <br>
   * takes the same arguments as the train() function in learning.cc file
   * \param args arguments passed to the command
  */
  void train(std::vector<std::string> args);
  /**
   * adds a layer to the network <br>
   * calls the corresponding network mlp method
   * necessarily the number of neurons and activation. Optionally passed index for insertion
   * \param args arguments passed to the command
  */
  void add_layer(std::vector<std::string> args);
  /**
   * calls the test function with the passed parameters
   * takes the same arguments as the test() function in learning.cc file
   * \param args arguments passed to the command
  */
  void test(std::vector<std::string> args);
  /**
   * Classifies the transmitted image using the current network <br>
   * necessarily requires the path to the image in .png format for which you need to make a prediction
   * \param args arguments passed to the command
  */
  void predict(std::vector<std::string> args);
  /**
   * loads the network from a file <br>
   * calls the corresponding model processing method <br>
   * requires one parameter - file path
   * \param args arguments passed to the command
  */
  void load(std::vector<std::string> args);

  void make(std::vector<std::string> args);

  /**
   * Returns a pointer to the corresponding method given the command name
   * \param command command for which you need to find a method
   * \return pointer to the corresponding method
  */
  void(Console::*get_command(std::string command))(std::vector<std::string>);

};



#endif