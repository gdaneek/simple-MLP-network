#ifndef MODSAVER_HH
#define MODSAVER_HH
#include "../include/mlp.hh"
#include <iostream>
#include <vector>
#include <map>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include<set>
#include <stdexcept>
#include <initializer_list>

// Правило: пользователь может сам создавать функции активации и делать с ними все что угодно, но если модуль подключен, функции активации обьявляются сами
// Модел сейверу необходимо знать информацию и иметь доступ к каждой функции активации, чтобы собирать разные сети, поэтому так





// size_t make_id(std::string str);
// activation_ptr get_ptr_by_id(size_t id);
// activation_ptr get_ptr_by_name(std::string activation_name);
// std::string get_name_by_id(size_t id);
// std::string get_name_by_ptr(activation_ptr ptr);
// size_t get_id_by_ptr(activation_ptr ptr);


struct ModelSaver {
    const static inline size_t file_signature = 0x10454c4946504c4d,
                               net_info_label = 0x104f464e4954454e;
    std::string make_filename(NetMLP &net);
    void save_net_to_file(NetMLP &net, std::string path="", std::string msg = "");
    
    void check_file_signature(std::ifstream& fin);
    static std::tuple<bool, NetMLP> netmaker(std::vector<std::pair<size_t, activations::fptr>>& layers, weight_vector& weights, std::vector<double>& shifts);

    std::tuple<bool, NetMLP, std::string> upload_net_from_file(std::string path);
    void show_model_info(NetMLP& net, std::string msg = "");
};

#endif