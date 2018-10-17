#include "relu_layer.h"

ReLU_Layer::ReLU_Layer(const QString &name, short inputHeight, short inputWidth, short inputDepth)
    : Layer(name)
{
    m_InputHeight = inputHeight;
    m_InputWidth = inputWidth;
    m_InputDepth = inputDepth;
}

void ReLU_Layer::Forward(arma::cube& input, arma::cube& output)
{
    output = arma::zeros(arma::size(input));
    output = arma::max(input, output);
}
