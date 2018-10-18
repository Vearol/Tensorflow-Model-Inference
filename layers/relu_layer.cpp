#include "relu_layer.h"

ReLU_Layer::ReLU_Layer(const QString &name)
    : Layer(name)
{
}

void ReLU_Layer::Forward(arma::cube& input, arma::cube& output)
{
    output = arma::zeros(arma::size(input));
    output = arma::max(input, output);
}
