#include "../include/modsaver.hh"

struct ActivationNamesMap {
    std::map<std::string, activations::ActivationFunc*> table;
    size_t make_id(std::string str) {
        size_t x{137}, res{0};
        for(size_t i{0}, x_{1};i < str.length();res = res*x_ + str[i], i++, x_ *= x);
        return res;
    }
    ActivationNamesMap() {
        this->table =  {
                                                                    std::make_pair("empty", &empty),      
                                                                    std::make_pair("relu", &relu),
                                                                    std::make_pair("tanh", &tanh_),
                                                                    std::make_pair("sigmoid", &sigmoid),
                                                                    std::make_pair("softmax", &softmax)
                                                            };
    }
    activations::ActivationFunc* get_ptr_by_id(size_t id) {
        for(auto elem : table)
            if(make_id(elem.first) == id) return elem.second;
        return nullptr;
    }
    std::string get_name_by_id(size_t id) {
        for(auto elem : table)
            if(make_id(elem.first) == id) return elem.first;
        return "";
    }
    std::string get_name_by_ptr(activations::ActivationFunc* ptr) {
        for(auto elem : table)
            if(elem.second == ptr) return elem.first;
        return "";
    }
    size_t get_id_by_ptr(activations::ActivationFunc* ptr) {
        for(auto elem : table)
            if(elem.second == ptr) return make_id(elem.first);
        return 0;
    }
};
struct ModelSaver {
    const static inline size_t file_id = 0x10454c4946504c4d,
                               net_info_id = 0x104f464e4954454e;
    ActivationNamesMap activ_names_map;

    std::string make_filename(Net &net) {
        std::string filename{"model"};
        for(Layer layer : net) filename += "-"+std::to_string(layer.size());    
        return filename+".mlp";
    }

    ModelSaver() : activ_names_map{ActivationNamesMap()} {};
    void save_net_to_file(Net &net, std::string path="", std::string msg = "") {
        std::ofstream fout(((path == "")?  make_filename(net) : path), std::ios::binary|std::ios::out);
        auto byte_writer{[](auto value, std::ofstream& f) {f.write((char*)&value, sizeof(value));}};
        auto weight_count{[](Net &net) {
            for(size_t i{0}, res{0};i <= net.NeuralLinks.size();
                res+=net.NeuralLinks[i].size()*net.NeuralLinks[i][0].size(), i++)
            if(i == net.NeuralLinks.size()) return res;
        }};
        for(size_t data : std::vector<size_t>{file_id, net.size(), weight_count(net)})byte_writer(data, fout);
        for(size_t i{0};i < net.size();i++) {
            byte_writer((size_t)net[i].size(), fout);
            byte_writer((size_t)activ_names_map.get_id_by_ptr(net.activations[i]), fout);
        }
        for(auto link : net.NeuralLinks) 
            for(auto weights : link) 
                for(auto weight : weights) 
                    byte_writer(weight, fout);
        for(auto layer : net) 
            for(auto neuron : layer)
                byte_writer(neuron.shift, fout);
        byte_writer(net_info_id, fout);
        for(auto c : msg)byte_writer(c, fout);
        byte_writer('\0', fout);
        fout.close();
    }
    
    void check_file_id(std::ifstream& fin) {
        size_t id;fin.read((char*)&id, sizeof(id));
        if(id != file_id)
            throw std::runtime_error{"This file is not an MLP model file"};
    }

    std::tuple<bool, Net> netmaker(std::vector<std::pair<size_t, activations::ActivationFunc*>>& layers, weight_vector& weights, std::vector<double>& shifts) {
        Net net(layers[0].first, (*--layers.end()).first, layers[0].second, (*--layers.end()).second);
        for(size_t i{1};i < layers.size()-1;i++) net.add(layers[i].first, layers[i].second);
        try {net.make();}    
        catch(...) {
            std::cout << "Model builder: This model uses invalid layers. Build failed\n";
            return {false, net};
        }
        try {
            for(size_t i{0}, w_iter{0};i < net.NeuralLinks.size();i++) 
                for(size_t j{0};j < net.NeuralLinks[i].size();j++) 
                    for(size_t g{0};g < net.NeuralLinks[i][j].size();g++) 
                        net.NeuralLinks[i][j][g] = weights[w_iter++];
            for(size_t i{0}, s_iter{0};i < net.size();i++) 
                for(size_t j{0};j < net[i].size();j++)
                    net[i][j].shift = shifts[s_iter++];
        }
        catch(...) {
            std::cout << "Model builder: missing weights or offsets. Initialization of trainable parameters failed\n";
            return {false, net};
        }
        return {true, net};
    }

    std::tuple<bool, Net, std::string> upload_net_from_file(std::string path) {
        std::ifstream fin(path, std::ios::binary|std::ios::in);
        auto byte_reader{[](auto &value, std::ifstream& f){f.read((char*)&value, sizeof(value));return value;}};
        check_file_id(fin);
        size_t layers_num, weight_num, shifts_num{0};
        for(auto data : std::vector<size_t*>{&layers_num, &weight_num})byte_reader(*data, fin);
        std::vector<std::pair<size_t, activations::ActivationFunc*>> layers(layers_num); // число нейронов + активация
        for(size_t i{0};i < layers_num;shifts_num += layers[i].first, i++) {
            size_t neurons_num, activation;
            byte_reader(neurons_num, fin);
            byte_reader(activation, fin);
            layers[i] = std::make_pair(neurons_num, activ_names_map.get_ptr_by_id(activation));
        }
        weight_vector weights(weight_num);
        for(size_t i{0};i < weight_num;i++) byte_reader(weights[i] , fin);
        std::vector<double> shifts(shifts_num);
        for(size_t i{0};i < shifts_num;i++) byte_reader(shifts[i], fin);
        size_t check_if_valid; byte_reader(check_if_valid, fin);
        if(check_if_valid != net_info_id)
            throw std::runtime_error{"error reading file"};
        std::string info{""};
        for(char x;fin;info+=byte_reader(x, fin));
        fin.close();
        auto [status, net] = netmaker(layers, weights, shifts);
        return {status, net, info};
    }
    void net_info(Net& net) {
        std::cout << "NET INFO: " << std::endl;
        for(size_t i{0};i < net.size();i++)
            std::cout << "\tLayer " << i << ": " << net[i].size() << " neurons with " << activ_names_map.get_name_by_ptr(net.activations[i]) << std::endl;
        std::cout << "Number of layers: " << net.size() << std::endl;
    }
};

// Eaxmple usage
/*
int main() {
    std::srand(std::time(nullptr));
    ModelSaver mdls;
    Net net(12,15,&empty, &softmax);
    net.add(30, &relu);
    net.add(5, &sigmoid);
    net.make();
    mdls.save_net_to_file(net);
    auto [status, net_u, msg] = mdls.upload_net_from_file("model-12-30-5-15.mlp");
    
    mdls.net_info(net_u);
    std::cout << msg << std::endl;
    return 0;
}
*/