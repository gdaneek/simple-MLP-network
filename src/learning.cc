#include "../include/learning.hh"

const int count_of_figures = 5;

std::vector<neuron_t> vectorize_image(std::string fpath) {
    sf::Image img;
    img.loadFromFile(fpath);
    auto [rows, cols] = img.getSize();
    std::vector<neuron_t> res(rows*cols);
    for(size_t i{0};i < rows;i++) {
        for(size_t j{0};j < cols;j++) {
            res[i*rows+j] = (img.getPixel(i,j).r > 0);
        }
    }
    return res;
}

template<typename LabelType>
std::vector<neuron_t> vectorize_lable(LabelType& label) {
    std::vector<neuron_t> res;
    for (size_t i{0}; i < count_of_figures; i++) {
        (i == label) ? res.push_back(1) : res.push_back(0);
    }
    return res;
}

void NetMLP::backprop(const std::vector<double>& target, double learning_rate) {
    size_t count_ones{0};
    for (size_t i{0}; i < target.size(); i++) {
        if (target[i] == 1) {
            count_ones++;
        }
    }
    if (count_ones != 1) {
        throw std::runtime_error{"unsuitable target"};
    }

    // calc err---------------------
    double mse{0};
    for (size_t j{0}; j < layers.back().size(); j++) {
        mse += (target[j] - layers.back()[j].get_value()) * (target[j] - layers.back()[j].get_value());
    }
    mse /= layers.back().size();
    std::cout << mse << std::endl;
    //--------------------------------
    
    size_t last = layers.size() - 1;
    std::vector<weight> delta_previous(layers.back().size());

    for (size_t j{0}; j < layers.back().size(); j++) {
        delta_previous[j] = (target[j] - layers.back()[j].get_value()) * activations::derivative(layers.back()[j], *activations[last], 1e-6);
    }

    // last layer shift changes
    for (size_t j{0}; j < layers.back().size(); j++) {
        layers.back()[j].shift += delta_previous[j] * learning_rate;
    }


    std::vector<weight> delta;
    for (int j{last - 1}; j >= 0; j--) {
        delta.resize(layers[j].size());
        // find d
        for (size_t k{0}; k < layers[j].size(); k++) {
            double sum_d = 0;
            for (size_t t{0}; t < delta_previous.size(); t++) {
                sum_d += delta_previous[t] * links[j].get_weight(t, k);
            }
            delta[k] = sum_d * activations::derivative(layers[j][k], *activations[j], 1e-6);
        }
        // change w with delta_previous (from k to t) 
        for (size_t k{0}; k < layers[j].size(); k++) {
            layers[j][k].shift += delta[k] * learning_rate;
            for (size_t t{0}; t < delta_previous.size(); t++) {
                double dw = delta_previous[t] * layers[j][k].get_value() * learning_rate; 
                weight new_weight = links[j].get_weight(t, k) + dw;
                links[j].set_weight(t, k, new_weight);
            }
        }
        // delta_previous = delta
        delta_previous.clear();
        delta_previous.resize(delta.size());
        delta_previous = delta;
        delta.clear();

    }
}

void train(NetMLP& net, std::string dir, double learning_rate) {
    std::set<std::string> shapes, lables;
    for (const auto& entry : fs::directory_iterator(dir)) {
        std::string path = entry.path();
        if(path.find("shape") != std::string::npos)shapes.insert(path);
        else lables.insert(path);
    }
    if(shapes.size() != lables.size())
        throw std::runtime_error{"The number of labels is not equal to the number of images\n"};
  
    for(auto shapes_it{shapes.begin()}, lables_it{lables.begin()};lables_it != lables.end();shapes_it++, lables_it++) {
     
        auto input = vectorize_image(*shapes_it);
        std::ifstream fin(*lables_it);
        int lable;
        fin >> lable;
        auto target = vectorize_lable<int>(lable);
        auto res = net.feedforward(input);
        std::cout << std::get<size_t>(res) << " " << std::get<neuron_t>(res)<< std::endl;

        net.backprop(target, learning_rate);
    }
}

std::pair<size_t, size_t> test(NetMLP& net, std::string dir) {
    std::set<std::string> shapes, labels;
    for (const auto& entry : fs::directory_iterator(dir)) {
        std::string path = entry.path();
        if(path.find("shape") != std::string::npos)shapes.insert(path);
        else labels.insert(path);
    }
    if(shapes.size() != labels.size())
        throw std::runtime_error{"The number of labels is not equal to the number of images\n"};

    std::pair<size_t, size_t> test_results{0, 0};       // test, success tests

    for(auto shapes_it{shapes.begin()}, labels_it{labels.begin()};labels_it != labels.end();shapes_it++, labels_it++) {
     
        auto input = vectorize_image(*shapes_it);
        std::ifstream fin(*labels_it);
        int label;
        fin >> label;
        auto target = vectorize_lable<int>(label);
        auto res = net.feedforward(input);
        std::cout << std::get<size_t>(res) << " " << std::get<neuron_t>(res)<< std::endl;

        test_results.first += 1;
        if (label == std::get<size_t>(res)) {
            test_results.second += 1;
        }
    }
    return test_results;
}