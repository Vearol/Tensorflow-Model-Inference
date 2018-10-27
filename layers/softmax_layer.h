#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include "functional_layer.h"

class Softmax_Layer : public Functional_layer
{
public:
    Softmax_Layer(const QString &name);

    void forward(arma::vec &input, arma::vec &output) override;
};

#endif // SOFTMAX_LAYER_H
