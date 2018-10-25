#include "relu_layer.h"

ReLU_Layer::ReLU_Layer(const QString &name) : Functional_layer(name)
{
}

void ReLU_Layer::Forward(arma::cube& input, arma::cube& output)
{
    output = arma::zeros(arma::size(input));
    output = arma::max(input, output);
}

void ReLU_Layer::Forward(arma::vec &input, arma::vec &output)
{
    output = arma::zeros(arma::size(input));
    output = arma::max(input, output);
}
