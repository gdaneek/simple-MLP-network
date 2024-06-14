#include "SFML/Graphics.hpp"
#include "console.hh"
#include "learning.hh"
#include "mlp.hh"
#include "stdarg.h"
#include "top/activations.hh"
#include "top/net_ms.hh"

std::vector<std::string> split(const std::string& str,
                               const std::string& delimiter) {
  std::vector<std::string> tokens;
  size_t start = 0;
  size_t end = str.find(delimiter);
  while (end != std::string::npos) {
    tokens.push_back(str.substr(start, end - start));
    start = end + delimiter.length();
    end = str.find(delimiter, start);
  }
  tokens.push_back(str.substr(start));
  return tokens;
}

bool process_script(Console& console, std::string path) {
  std::ifstream fin(path);
  if (!fin.is_open()) {
    console.println("Unable to open the script file");
    return true;
  }

  size_t line_it{0};
  while (!fin.eof()) {
    std::string input;
    std::getline(fin, input);
    if (input == "") continue;
    if ((input == "q") | (input == "exit")) break;
    std::vector<std::string> splitted = split(input, " ");
    auto command = console.get_command(*splitted.begin());
    splitted.erase(splitted.begin());
    if (command != nullptr)
      (console.*command)(splitted);
    else {
      console.println("Unknown command at line " + std::to_string(line_it) +
                      ". Exiting");
      return true;
    }
    ++line_it;
  }
  fin.close();
  return false;
}

int main(int argc, char* argv[]) {
  std::srand(std::time(nullptr));
  Console console(std::cout);

  if ((argc > 2) && ((std::string)argv[1] == "-S"))
    return process_script(console, argv[2]);

  while (1) {
    std::string progname{std::string(argv[0])};
    std::cout << progname.substr(progname.rfind("/") + 1) << "> ";
    std::string input;
    std::getline(std::cin, input);
    if ((input == "q") | (input == "exit")) break;
    std::vector<std::string> splitted = split(input, " ");
    auto command = console.get_command(*splitted.begin());
    splitted.erase(splitted.begin());
    if (command != nullptr)
      (console.*command)(splitted);
    else
      console.println("Unknown command");
  }

  return 0;
}
