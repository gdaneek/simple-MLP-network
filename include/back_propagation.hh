#ifndef BACK_PROPAGATION_HH
#define BACK_PROPAGATION_HH
#include <vector>
#include <cmath>
#include "mlp.hh"

void forward_propagation(Net& net, weight_vector& x);

void back_propagation(double n, double a, Net& net, weight_vector& x, std::vector<double>& t, int steps);

#endif 