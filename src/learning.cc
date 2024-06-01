
#include "../include/learning.hh"


std::vector<neuron_t> vectorize_image(std::string fpath) {
    sf::Image img;
    img.loadFromFile(fpath);
    auto [rows, cols] = img.getSize();
    std::vector<neuron_t> res(rows*cols);
    for(size_t i{0};i < rows;i++)
        for(size_t j{0};j < cols;j++)
            res[i*rows+j] = (img.getPixel(i,j).r > 0);
    return res;
}

template<typename LabelType>
std::vector<neuron_t> vectorize_label(LabelType& label) {
    std::vector<neuron_t> res;
    for(size_t i{0};i < (sizeof(LabelType)<<3);i++)
        res.push_back((label&(1ULL << i)) > 0);
    return res;
}
void train(NetMLP& net, std::string dir) {
  
    std::set<std::string> shapes, labels;
    for (const auto& entry : fs::directory_iterator(dir)) {
        std::string path = entry.path();
        if(path.find("shape") != std::string::npos)shapes.insert(path);
        else labels.insert(path);
    }
    if(shapes.size() != labels.size())
        throw std::runtime_error{"The number of labels is not equal to the number of images\n"};
  
  
    for(auto shapes_it{shapes.begin()}, labels_it{labels.begin()};labels_it != labels.end();shapes_it++, labels_it++) {
     
        auto input = vectorize_image(*shapes_it);
        std::ifstream fin(*labels_it);
        char label;
        fin >> label;
        auto target = vectorize_label<char>(label);
       
        auto res = net.feedforward(input);
        std::cout << std::get<size_t>(res) << " " << std::get<neuron_t>(res)<< std::endl;
        net.backprop(target, 0.5);
    }
}
std::tuple<size_t, size_t> test(NetMLP& net, std::string dir) { // ОБЩЕЕ ЧИСЛО ТЕСТОВ И СКОЛЬКО ИЗ НИХ ПРОЙДЕНЫ УСПЕШНО
    ;
}