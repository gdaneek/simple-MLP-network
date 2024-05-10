#ifndef BACK_PROPAGATION_HH
#define BACK_PROPAGATION_HH
#include <vector>
#include <cmath>
#include "mlp.hh"

void back_propagation(double n, double a, std::vector<std::pair<std::vector<int>, int>>& x_t, int steps);


#endif 