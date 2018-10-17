#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "layer.h"

class Dense_Layer : public Layer
{

private:
    short m_InputHeight;
    short m_InputWidth;
    short m_InputDepth;
    arma::cube m_Input;

    short m_NumOutputs;
    arma::vec m_Output;

    arma::mat m_Weights;
    arma::vec m_Biases;

public:
    Dense_Layer(
            const QString &name,
            short inputHeight,
            short inputWidth,
            short inputDepth,
            short numOutputs);

    virtual void Forward(arma::cube &input, arma::vec &output);
};

#endif // DENSE_LAYER_H
