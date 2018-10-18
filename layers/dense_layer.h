#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "layer.h"

class Dense_Layer : public Layer
{

private:
    arma::cube m_Input;

    arma::mat m_Weights;
    arma::vec m_Biases;

public:
    Dense_Layer(const QString &name);

    void Forward(arma::vec &input, arma::vec &output) override;
    void Initialize_Weights(const QStringList &text) override;
    void Initialize_Biases(const QStringList &text) override;
};

#endif // DENSE_LAYER_H
