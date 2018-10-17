#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include "layer.h"

class Softmax_Layer : public Layer
{
public:
    Softmax_Layer(const QString &name);

    void Forward(arma::vec &input, arma::vec &output);
};

#endif // SOFTMAX_LAYER_H
