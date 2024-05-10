#include "../include/back_propagation.hh"
// #include "../include/mlp.hh"

void forward_propagation(Net& net, weight_vector& x) {
    net.set_input(x);
    for (size_t i{1}; i < net.size(); i++) {
        net[i] = net[i - 1] * net.NeuralLinks[i - 1];
        net[i].apply_offsets();
        (*net.activations[i])(net[i]);
    }
}

void back_propagation(double n, double a, Net& net, weight_vector& x, std::vector<double>& t, int steps) {
    for (int i{0}; i < steps; i++) {
        forward_propagation(net, x);
        size_t last = net.size() - 1;
        weight_vector d_previous(net[last].size());
        for (size_t j{0}; j < net[last].size(); j++) {
            // d_previous[i] = -net[last][j].value * (1 - net[last][j].value) * (t[i] - net[last][j].value);  // rec func ???
            d_previous[i] = -2 * a * net[last][j].value * (1 - net[last][j].value) * (t[i] - net[last][j].value);
        }
        weight_vector d;
        for (size_t j{last - 1}; j >= 0; j--) {
            d.resize(net[j].size());
            // find d
            for (size_t k{0}; k < net[j].size(); k++) {
                double sum_d = 0;
                for (size_t t{0}; t < d_previous.size(); t++) {
                    sum_d += d_previous[t] * net.NeuralLinks[j].weights[k][t];
                }
                d[k] = 2 * a * net[k][j].value * (1 - net[k][j].value) * sum_d;
            }
            // change w with d_previous (from k to t)
            for (size_t k{0}; k < net[j].size(); k++) {
                for (size_t t{0}; t < net.NeuralLinks.size(); t++) {
                    double dw = -n * d_previous[t] * net[j][k].value; 
                    net.NeuralLinks[j].weights[k][t] += dw;     // forgot bias changes
                }
            }
            // d_previous = d
            d_previous.clear();
            d_previous.resize(d.size());
            d_previous = d;
            d.clear();
        }
    }
}