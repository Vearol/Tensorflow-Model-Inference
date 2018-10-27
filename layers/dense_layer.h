#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "physical_layer.h"

class Dense_Layer : public Physical_Layer
{

private:
    arma::mat m_Weights;

    void initialize_weights(const std::string &array_path) override;

public:
    Dense_Layer(const QString &name);

    void forward(arma::vec &input, arma::vec &output) override;

};

#endif // DENSE_LAYER_H
