#ifndef LAYER_H
#define LAYER_H

#include <armadillo>
#include <QString>

class Layer
{
protected:
    QString m_Name;        

public:
    Layer(const QString &name);

    virtual ~Layer();

    virtual void forward(arma::cube &input, arma::cube &output);
    virtual void forward(arma::vec &input, arma::vec &output);

    virtual void initialize_weights_and_Biases(const QString &layer_directory_path);

    QString get_name();
};

#endif // LAYER_H
