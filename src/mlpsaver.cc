#include "../include/mlp.hh"
#include "../include/top/activations.hh"

const std::string MLPModelSaver::make_filename(NetMLP& net) {
    std::string filename{"model"};
    for(auto& layer : net.layers) 
        filename += "-"+std::to_string(layer.size());    
    return filename+".mlp";
}

void MLPModelSaver::check_file_signature(std::ifstream& fin) const {
    size_t bytes;
    fin.read((char*)&bytes, sizeof(bytes));
    if(bytes != file_signature)
        throw std::runtime_error{"Model builder: This file is not an MLP model file"};
}


std::string MLPModelSaver::save_net_to_file(NetMLP &net, std::string path, std::string msg) {
    std::ofstream fout(((path == "")?  make_filename(net) : path), std::ios::binary|std::ios::out);
    auto byte_writer{[](auto value, std::ofstream& f) {f.write((char*)&value, sizeof(value));}};
    auto weight_count{[](NetMLP &net) {
        size_t res{0};
        for(size_t i{0};i < net.links.size();res+=net.links[i].size(), i++);
        return res;
    }};
    
    for(uint64_t data : std::vector<uint64_t>{file_signature, net.size(), weight_count(net)})
        byte_writer(data, fout);

    for(size_t i{0};i < net.size();i++) {
        byte_writer((uint64_t)net.layers[i].size(), fout);
        byte_writer((uint64_t)activations::table.id_by_ptr(net.activations[i]), fout);
    }

    for(auto link : net.links) 
        for(auto weight_v : link) 
            byte_writer(weight_v, fout);
          
    for(auto layer : net.layers) 
        //for(auto neuron : layer)
        for(size_t i{0};i < layer.size();i++) 
            byte_writer(static_cast<NeuronMLP>(layer[i]).shift, fout);
    byte_writer(net_info_label, fout);
    for(char c : msg)
        byte_writer(c, fout);
    byte_writer('\0', fout);
    fout.close();
    return (path == "")?  make_filename(net) : path;
}


NetMLP MLPModelSaver::netmaker(std::vector<std::tuple<size_t, activations::fptr>>& layers, weight_vector& weights, std::vector<neuron_t>& shifts) {
    NetMLP net(std::get<size_t>(*layers.begin()), std::get<activations::fptr>(*layers.begin()),
               std::get<size_t>(*--layers.end()), std::get<activations::fptr>(*--layers.end()));

    for(size_t i{1};i < layers.size()-1;i++) 
        net.add(std::get<size_t>(layers[i]), std::get<activations::fptr>(layers[i]));
    try {
            net.make();
    }    
    catch(...) {
        throw std::runtime_error{"Model builder: This model uses invalid layers. Build failed\n"};
    }
    try {
        for(size_t i{0}, w_iter{0};i < net.links.size();i++) 
            for(size_t j{0};j < net.links[i].size();j++) 
                    net.links[i].weights[j] = weights[w_iter++];

        for(size_t i{}, s_iter{};i < net.size();i++) 
            for(size_t j{};j < net.layers[i].size();j++) 
                net.layers[i][j].shift = shifts[s_iter++];
    }
    catch(...) {
        throw std::runtime_error{"Model builder: missing weights or offsets. Initialization of trainable parameters failed\n"};
    }
    return net;
}

std::tuple<NetMLP, std::string> MLPModelSaver::upload_net_from_file(std::string path) {
    std::ifstream fin(path, std::ios::binary|std::ios::in);
    auto byte_reader{[](auto &value, std::ifstream& f){f.read((char*)&value, sizeof(value));return value;}};
    check_file_signature(fin);
    uint64_t layers_num, weight_num, shifts_num{0};
    for(auto data : std::vector<uint64_t*>{&layers_num, &weight_num})
        byte_reader(*data, fin);
    
    std::vector<std::tuple<size_t, activations::fptr>> layers(layers_num); // число нейронов + активация

    for(size_t i{0};i < layers_num;shifts_num += std::get<size_t>(layers[i]), i++) {
        uint64_t neurons_num, activation;
        byte_reader(neurons_num, fin);
        byte_reader(activation, fin);
        try {
            layers[i] = std::make_pair(neurons_num, activations::table.get_by_id(activation));
        }
        catch(std::runtime_error& err) {
            std::cout << "Model builder: " << err.what() << std::endl;
            throw;
        }
    }

    weight_vector weights(weight_num);
    for(size_t i{0};i < weight_num;i++) 
        byte_reader(weights[i] , fin);

    std::vector<neuron_t> shifts(shifts_num);
    for(size_t i{0};i < shifts_num;i++) 
        byte_reader(shifts[i], fin);
    

    uint64_t check_valid;
    byte_reader(check_valid, fin);
    if(check_valid != net_info_label)
        throw std::runtime_error{"Model builder: This file was damaged or saved incorrectly"};
    std::string info{""};
    char x;
    for(;;) {
        x = byte_reader(x, fin);
        if(x == '\0')break;
        info += x;
    }

    fin.close();
    return {netmaker(layers, weights, shifts), info};
}


void MLPModelSaver::show_model_info(NetMLP& net, std::string msg, std::ostream& ostream) const{
    ostream << "\033[95mNet information: \033[39m\n";
    ostream << "\033[96mNumber of layers: \033[39m" << net.size() << std::endl;
    for(size_t i{0};i < net.size();i++)
        ostream << "\t\033[92mLayer " << i << ": \033[39m" << net.layers[i].size() << " neurons with " << activations::table.name_by_ptr(net.activations[i]) << std::endl;
    ostream << "\033[96mMessage from file:\033[39m" << ((msg == "")? "" : "\n"+msg) << std::endl;
}