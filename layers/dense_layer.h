#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "layer.h"

class Dense_Layer : public Layer
{

private:
    arma::mat m_Weights;
    arma::vec m_Biases;

    void Initialize_Weights(const QString &array_path) override;
    void Initialize_Biases(const QString &array_path) override;

public:
    Dense_Layer(const QString &name);

    void Forward(arma::vec &input, arma::vec &output) override;

};

#endif // DENSE_LAYER_H
