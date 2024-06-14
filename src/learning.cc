#include "../include/learning.hh"

#include "../include/mlp.hh"        // delete after tests, or not
#include "../include/top/activations.hh"


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
    // for(size_t i{0};i < (sizeof(LabelType)<<3);i++)
    //     // res.push_back((label&(1ULL << i)) > 0);     // change to get int
    //     res.push_back(label);                           // maybe change label type to int (neuronval)
    int count_of_figures = 7;   // take from somewhere!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for (size_t i{0}; i < count_of_figures; i++) {
        (i == label) ? res.push_back(1) : res.push_back(0);
    }
    return res;
}

// void NetMLP::backprop(const std::vector<double>& target, double learning_rate) {
//     size_t last = layers.size() - 1;
//     weight_vector delta_previous(layers.back().size());
//     for (size_t j{0}; j < layers.back().size(); j++) {
//         delta_previous[j] = -2 * learning_rate * layers.back()[j].get_value() * (1 - layers.back()[j].get_value()) * (target[j] - layers.back()[j].get_value());         // j ????
//     }
//     weight_vector delta;
//     for (size_t j{last - 1}; j >= 1; j--) {     // changed to j >= 1
//         delta.resize(layers[j].size());
//         // find d
//         for (size_t k{0}; k < layers[j].size(); k++) {
//             double sum_d = 0;
//             for (size_t t{0}; t < delta_previous.size(); t++) {
//                 sum_d += delta_previous[t] * links[j].get_weight(k, t);
//             }
//             delta[k] = 2 * learning_rate * layers[j][k].get_value() * (1 - layers[j][k].get_value()) * sum_d;
//         }
//         // change w with delta_previous (from k to t) 
//         for (size_t k{0}; k < layers[j].size(); k++) {
//             // layers[j][k].shift -= delta_previous[k];                       // shift changes
//             layers[j][k].shift -= delta_previous[k]; 
//             for (size_t t{0}; t < links.size(); t++) {
//                 double dw = delta_previous[t] * layers[j][k].get_value(); 
//                 weight new_weight = links[j].get_weight(k, t) - dw;
//                 links[j].set_weight(k, t, new_weight);
//                 // links[j].weights[k][t] += dw;
//             }
//         }
//         // delta_previous = delta
//         delta_previous.clear();
//         delta_previous.resize(delta.size());
//         delta_previous = delta;
//         delta.clear();
//     }
// }


// void NetMLP::backprop(const std::vector<double>& target, double learning_rate) {
//     size_t last = layers.size() - 1;
//     std::vector<weight> delta_previous(layers.back().size());
//     for (size_t j{0}; j < layers.back().size(); j++) {
//         delta_previous[j] = (-1) * layers.back()[j].get_value() * (1 - layers.back()[j].get_value()) * (target[j] - layers.back()[j].get_value());
//     }
//     std::vector<weight> delta;
//     for (size_t j{last - 1}; j >= 1; j--) {     // changed to j >= 1
//         // // print all weights 
//         // std::ofstream fout("./outputWeights.txt");
//         // for (size_t q{0}; q < layers.size(); q++) {n

//         //     for (size_t r{0}; r )
//         // }
//         delta.resize(layers[j].size());
//         // find d
//         for (size_t k{0}; k < layers[j].size(); k++) {
//             double sum_d = 0;
//             for (size_t t{0}; t < delta_previous.size(); t++) {
//                 sum_d += delta_previous[t] * links[j].get_weight(k, t);
//             }
//             delta[k] = layers[j][k].get_value() * (1 - layers[j][k].get_value()) * sum_d;
//         }
//         // change w with delta_previous (from k to t) 
//         for (size_t k{0}; k < layers[j].size(); k++) {
//             // layers[j][k].shift -= delta_previous[k];                       // shift changes
//             // layers[j][k].shift -= delta_previous[k] * learning_rate;
//             layers[j][k].shift -= delta[k] * learning_rate; 
//             for (size_t t{0}; t < links.size(); t++) {
//                 // double dw = delta_previous[t] * layers[j][k].get_value() * learning_rate; 
//                 double dw = delta[k] * layers[j][k].get_value() * learning_rate; 
//                 weight new_weight = links[j].get_weight(k, t) - dw;
//                 links[j].set_weight(k, t, new_weight);
//                 // links[j].weights[k][t] += dw;
//             }
//         }
//         // delta_previous = delta
//         delta_previous.clear();
//         delta_previous.resize(delta.size());
//         delta_previous = delta;
//         delta.clear();

//     }
// }

// void NetMLP::backprop(const std::vector<double>& target, double learning_rate) {
//     size_t last = layers.size() - 1;
//     std::vector<weight> delta_previous(layers.back().size());
//     for (size_t j{0}; j < layers.back().size(); j++) {
//         delta_previous[j] = (-1) * layers.back()[j].get_value() * (1 - layers.back()[j].get_value()) * (target[j] - layers.back()[j].get_value());
//     }
//     std::vector<weight> delta;
//     for (size_t j{last - 1}; j >= 1; j--) {     // changed to j >= 1
//         delta.resize(layers[j].size());
//         // find d
//         for (size_t k{0}; k < layers[j].size(); k++) {
//             double sum_d = 0;
//             for (size_t t{0}; t < delta_previous.size(); t++) {
//                 sum_d += delta_previous[t] * links[j].get_weight(k, t);
//             }
//             delta[k] = layers[j][k].get_value() * (1 - layers[j][k].get_value()) * sum_d;
//         }
//         // change w with delta_previous (from k to t) 
//         for (size_t k{0}; k < layers[j].size(); k++) {
//             layers[j][k].shift -= delta[k] * learning_rate; 
//             for (size_t t{0}; t < delta_previous.size(); t++) {
//                 double dw = delta_previous[t] * layers[j][k].get_value() * learning_rate; 
//                 weight new_weight = links[j].get_weight(k, t) - dw;
//                 links[j].set_weight(k, t, new_weight);
//             }
//         }
//         // delta_previous = delta
//         delta_previous.clear();
//         delta_previous.resize(delta.size());
//         delta_previous = delta;
//         delta.clear();

//     }
// }


void NetMLP::backprop(const std::vector<double>& target, double learning_rate) {
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
        // delta_previous[j] = err * activations::derivative(layers.back()[j], *activations[last], 1e-6);
        delta_previous[j] = (target[j] - layers.back()[j].get_value()) * activations::derivative(layers.back()[j], *activations[last], 1e-6);
        // delta_previous[j] = (-1) * (target[j] - layers.back()[j].get_value()) * activations::derivative(layers.back()[j], *activations[last], 1e-6);
        // delta_previous[j] = (-1) * layers.back()[j].get_value() * (1 - layers.back()[j].get_value()) * (target[j] - layers.back()[j].get_value());
    }

    // last layer shift changes
    for (size_t j{0}; j < layers.back().size(); j++) {
        layers.back()[j].shift -= delta_previous[j] * learning_rate;
    }


    std::vector<weight> delta;
    for (int j{last - 1}; j >= 0; j--) {
        delta.resize(layers[j].size());
        // find d
        for (size_t k{0}; k < layers[j].size(); k++) {
            double sum_d = 0;
            for (size_t t{0}; t < delta_previous.size(); t++) {
                // sum_d += delta_previous[t] * links[j].get_weight(k, t);
                sum_d += delta_previous[t] * links[j].get_weight(t, k);
            }
            // delta[k] = layers[j][k].get_value() * (1 - layers[j][k].get_value()) * sum_d;
            delta[k] = sum_d * activations::derivative(layers[j][k], *activations[j], 1e-6);
            // delta[k] = sum_d * ;
            // delta[k] = (-1) * sum_d * activations::derivative(layers[j][k], *activations[j], 1e-6);
        }
        // change w with delta_previous (from k to t) 
        for (size_t k{0}; k < layers[j].size(); k++) {
            layers[j][k].shift -= delta[k] * learning_rate;
            for (size_t t{0}; t < delta_previous.size(); t++) {
                double dw = delta_previous[t] * layers[j][k].get_value() * learning_rate; 
                // weight new_weight = links[j].get_weight(k, t) - dw;
                // links[j].set_weight(k, t, new_weight);
                weight new_weight = links[j].get_weight(t, k) - dw;
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


// // for tangh
// void NetMLP::backprop(const std::vector<double>& target, double learning_rate, double moment) {
//     // calc err---------------------
//     double mse{0};
//     for (size_t j{0}; j < layers.back().size(); j++) {
//         mse += (target[j] - layers.back()[j].get_value()) * (target[j] - layers.back()[j].get_value());
//     }
//     mse /= layers.back().size();
//     std::cout << mse << std::endl;
//     //--------------------------------

//     size_t last = layers.size() - 1;
//     std::vector<weight> delta_previous(layers.back().size());

//     for (size_t j{0}; j < layers.back().size(); j++) {
//         delta_previous[j] = (target[j] - layers.back()[j].get_value()) * activations::derivative(layers.back()[j], *activations[last], 1e-6);
//         // delta_previous[j] = (-1) * layers.back()[j].get_value() * (1 - layers.back()[j].get_value()) * (target[j] - layers.back()[j].get_value());
//         // delta_previous[j] = layers.back()[j].get_value() * (1 - layers.back()[j].get_value()) * (target[j] - layers.back()[j].get_value());
//     }

//     // last layer shift changes
//     for (size_t j{0}; j < layers.back().size(); j++) {
//         layers.back()[j].shift -= delta_previous[j] * learning_rate;
//     }


//     std::vector<weight> delta;
//     for (int j{last - 1}; j >= 0; j--) {
//         delta.resize(layers[j].size());
//         // find d
//         for (size_t k{0}; k < layers[j].size(); k++) {
//             double sum_d = 0;
//             for (size_t t{0}; t < delta_previous.size(); t++) {
//                 sum_d += delta_previous[t] * links[j].get_weight(t, k);
//             }
//             delta[k] = sum_d * (1 - layers[j][k].get_value() * layers[j][k].get_value());
//         }
//         // change w with delta_previous (from k to t) 
//         for (size_t k{0}; k < layers[j].size(); k++) {
//             layers[j][k].shift -= delta[k] * learning_rate;
//             for (size_t t{0}; t < delta_previous.size(); t++) {
//                 double grad{delta_previous[t] * layers[j][k].get_value()};
//                 double dw = grad * learning_rate + moment * links[j].get_weight(t, k);
//                 weight new_weight = links[j].get_weight(t, k) - dw;
//                 links[j].set_weight(t, k, new_weight);
//             }
//         }
//         // delta_previous = delta
//         delta_previous.clear();
//         delta_previous.resize(delta.size());
//         delta_previous = delta;
//         delta.clear();

//     }
// }


void train(NetMLP& net, std::string dir) {
  
    std::set<std::string> shapes, labels;
    for (const auto& entry : fs::directory_iterator(dir)) {
        std::string path = entry.path();
        if(path.find("shape") != std::string::npos)shapes.insert(path);
        else labels.insert(path);
    }
    if(shapes.size() != labels.size())
        throw std::runtime_error{"The number of labels is not equal to the number of images\n"};
    // std::cout << "\n SHAPES: \n";
    // for(auto& s : shapes)std::cout << s << std::endl;
    // std::cout << "\n LABELS: \n";
  
    for(auto shapes_it{shapes.begin()}, labels_it{labels.begin()};labels_it != labels.end();shapes_it++, labels_it++) {
     
        auto input = vectorize_image(*shapes_it);
        std::ifstream fin(*labels_it);
        // char label;
        int label;          // change label to int
        fin >> label;
        // auto target = vectorize_label<char>(label);
        auto target = vectorize_label<int>(label);
        auto res = net.feedforward(input);
        std::cout << std::get<size_t>(res) << " " << std::get<neuron_t>(res)<< std::endl;

        // net.backprop(target, 0.9, 0.05);
        net.backprop(target, 0.05);

        //std::cout << *shapes_it << " " << *labels_it << std::endl;
    }
}
// std::tuple<size_t, size_t> test(NetMLP& net, std::string dir) { // ОБЩЕЕ ЧИСЛО ТЕСТОВ И СКОЛЬКО ИЗ НИХ ПРОЙДЕНЫ УСПЕШНО

std::pair<size_t, size_t> test(NetMLP& net, std::string dir) {
    std::set<std::string> shapes, labels;
    for (const auto& entry : fs::directory_iterator(dir)) {
        std::string path = entry.path();
        if(path.find("shape") != std::string::npos)shapes.insert(path);
        else labels.insert(path);
    }
    if(shapes.size() != labels.size())
        throw std::runtime_error{"The number of labels is not equal to the number of images\n"};
    // std::cout << "\n SHAPES: \n";
    // for(auto& s : shapes)std::cout << s << std::endl;
    // std::cout << "\n LABELS: \n";
    

    std::pair<size_t, size_t> test_results{0, 0};       // test, success tests

    for(auto shapes_it{shapes.begin()}, labels_it{labels.begin()};labels_it != labels.end();shapes_it++, labels_it++) {
     
        auto input = vectorize_image(*shapes_it);
        std::ifstream fin(*labels_it);
        // char label;
        int label;          // change label to int
        fin >> label;
        // auto target = vectorize_label<char>(label);
        auto target = vectorize_label<int>(label);
        auto res = net.feedforward(input);
        std::cout << std::get<size_t>(res) << " " << std::get<neuron_t>(res)<< std::endl;

        test_results.first += 1;
        if (label == std::get<size_t>(res)) {
            test_results.second += 1;
        }

        //std::cout << *shapes_it << " " << *labels_it << std::endl;
    }

    return test_results;
}