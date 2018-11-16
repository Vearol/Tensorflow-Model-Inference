#ifndef PHYSICAL_LAYER_H
#define PHYSICAL_LAYER_H

#include "layer.h"

class Physical_Layer : public Layer
{
public:
    Physical_Layer(const QString &name);

protected:
    arma::vec m_Biases;

    void initialize_weights_and_Biases(const QString &layer_directory_path) override;

    virtual void initialize_weights(const std::string &array_path) = 0;

private:
    void initialize_biases(const std::string &array_path);
};


#endif // PHYSICAL_LAYER_H