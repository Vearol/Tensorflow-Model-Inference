#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "layer.h"

class ReLU_Layer : public Layer
{
public:
    ReLU_Layer(const QString &name);

    void Forward(arma::cube &input, arma::cube &output) override;
};

#endif // RELU_LAYER_H
