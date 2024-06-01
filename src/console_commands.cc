#include "../include/console.hh"

std::map<std::string, void(Console::*)(std::vector<std::string>)> Console::commands {
    {"save", &Console::save},
    {"load", &Console::load},
    {"info", &Console::net_info},
    {"new", &Console::new_net},
    {"predict", &Console::predict},
    {"add", &Console::add_layer},
    {"make", &Console::make}
};