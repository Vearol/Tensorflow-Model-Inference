#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "functional_layer.h"

class ReLU_Layer : public Functional_layer
{
public:
    ReLU_Layer(const QString &name);

    void Forward(arma::cube &input, arma::cube &output) override;
};

#endif // RELU_LAYER_H
