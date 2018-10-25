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

    virtual void Forward(arma::cube &input, arma::cube &output);
    virtual void Forward(arma::vec &input, arma::vec &output);

    virtual void Initialize_Weights_and_Biases(const QString &layer_directory_path);

    QString Get_Name();
};

#endif // LAYER_H
