#include "softmax_layer.h"

Softmax_Layer::Softmax_Layer(const QString &name) : Functional_layer(name)
{
}

void Softmax_Layer::forward(arma::vec& input, arma::vec& output)
{
    double sumExp = arma::accu(arma::exp(input - arma::max(input)));
    output = arma::exp(input - arma::max(input))/sumExp;
}
