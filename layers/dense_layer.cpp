#include "dense_layer.h"

Dense_Layer::Dense_Layer(const QString &name, short inputHeight, short inputWidth, short inputDepth, short numOutputs)
    : Layer(name)
{
    m_InputHeight = inputHeight;
    m_InputWidth = inputWidth;
    m_InputDepth = inputDepth;
    m_NumOutputs = numOutputs;

    //todo load weights and biases
}

void Dense_Layer::Forward(arma::cube& input, arma::vec& output)
{
    arma::vec flatInput = arma::vectorise(input);
    output = (m_Weights * flatInput) + m_Biases;
}
