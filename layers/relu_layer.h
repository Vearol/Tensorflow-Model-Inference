#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "layer.h"

class ReLU_Layer : public Layer
{
private:
    short m_InputHeight;
    short m_InputWidth;
    short m_InputDepth;

public:
    ReLU_Layer(
            const QString &name,
            short inputHeight,
            short inputWidth,
            short inputDepth);

    void Forward(arma::cube &input, arma::cube &output);
};

#endif // RELU_LAYER_H
