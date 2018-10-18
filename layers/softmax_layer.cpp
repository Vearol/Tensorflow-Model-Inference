#include "softmax_layer.h"

Softmax_Layer::Softmax_Layer(const QString &name) : Layer(name)
{
}

void Softmax_Layer::Forward(arma::vec& input, arma::vec& output)
{
    double sumExp = arma::accu(arma::exp(input - arma::max(input)));
    output = arma::exp(input - arma::max(input))/sumExp;
}
