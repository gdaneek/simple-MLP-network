#include "../include/console.hh"

std::vector<std::tuple<size_t, std::string>> lnmap_reader(std::string dpath) {
    std::vector<std::tuple<size_t, std::string>> result;
    std::ifstream fin(dpath+"setup.lnmap");
    if(!fin.is_open())
        return std::vector<std::tuple<size_t, std::string>>{};
    uint8_t label_size = fin.get();
    uint8_t sep = fin.get();
    while(!fin.eof()) {
        size_t label = fin.get();
        if(fin.eof())break;
        std::string buff{""};
        for(char x = fin.get();x != sep;x = fin.get())
            buff += x;
        result.push_back({label, buff});
    }
    fin.close();
    return result;
}



void Console::print(std::string msg) {
    std::cout << msg;
}

void Console::println(std::string msg) {
    print(msg+"\n");
}

void Console::net_info(std::vector<std::string> args) {   //OK
    if(currNet == nullptr)return println("No net loaded");
    mds.show_model_info(*currNet);
}

void Console::save(std::vector<std::string> args) {      // OK
    if(currNet == nullptr)return println("No net loaded");
    auto [fpath, msg] = std::tuple((args.size())? *args.begin() : "",
                                    (args.size() > 1)? *(args.begin()+1) : "");
    auto _fpath = mds.save_net_to_file(*currNet,fpath, msg);
    println("Model saved to "+_fpath);
}

void Console::new_net(std::vector<std::string> args) {    // OK
    currNet.reset(new NetMLP());
    println("Empty net is initialized");
}

void Console::train(std::vector<std::string> args) {
    if(!args.size())return println(std::string(__func__)+" command requires at least 1 argument");
    std::string fpath = *args.begin();
    /* call train function... */
}

void Console::add_layer(std::vector<std::string> args) {
    if(args.size() < 2)
        return println(std::string(__func__)+" command requires at least 2 argument");
    try {
        size_t layer_sz = stoll(*args.begin());
        activations::fptr actf = activations::table.get_by_name(*++args.begin());
        currNet->add(layer_sz, actf);
        println("OK add layer");
    }
    catch(std::runtime_error& err) {
        std::string reason = err.what();
        println("Failed process command: "+reason);
    }
}

void Console::test(std::vector<std::string> args) {
    if(!args.size())
        return println(std::string(__func__)+" command requires at least 1 argument");
    std::string fpath = *args.begin();
    /* call test function... */
}

void Console::predict(std::vector<std::string> args) {// Предсказывает что изображено на картинке. 1 агрумент - картинка, второй - где можно поискать разметку
    if(currNet == nullptr)
        return println("No net loaded");
    auto img_path = *args.begin();
    auto input = vectorize_image(img_path);
    try { 
        //predict /home/danila/Documents/simple-MLP-network/src/dataset/train/shape_3_triangle.png /home/danila/Documents/simple-MLP-network/src/dataset/
        auto res = currNet->feedforward(input);
        auto lnmap = lnmap_reader((args.size() > 1)? *(args.begin()+1) : "");
        auto lbname = std::find_if(lnmap.begin(), lnmap.end(), [res](std::tuple<size_t, std::string> elem){
                                    return std::get<size_t>(elem) == std::get<size_t>(res);});
        auto classname = (lnmap.size()&&(lbname != lnmap.end()))? std::get<std::string>(*lbname) : std::to_string(std::get<size_t>(res));
        println("With "+std::to_string(std::get<neuronval>(res)*1e2).substr(0, 5)+"% probability it is "+classname+" class");
    }
    catch(std::runtime_error& err) {
        std::cout << err.what() << std::endl;
    }
}

void Console::load(std::vector<std::string> args) { //OK
    if(!args.size())return println(std::string(__func__)+" command requires at least 1 argument");
    std::string msg;
    auto uploaded = mds.upload_net_from_file(*args.begin());
    currNet = std::make_unique<NetMLP>(std::get<NetMLP>(uploaded));
    
}   

void(Console::*Console::get_command(std::string command))(std::vector<std::string>) {
    auto ptr = commands.find(command);
    return (ptr == commands.end())? nullptr : ptr->second;
}