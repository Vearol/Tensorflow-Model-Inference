#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "functional_layer.h"

class ReLU_Layer : public Functional_layer
{
public:
    ReLU_Layer(const QString &name);

    void forward(arma::cube &input, arma::cube &output) override;
    void forward(arma::vec &input, arma::vec &output) override;
};

#endif // RELU_LAYER_H
