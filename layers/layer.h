#ifndef LAYER_H
#define LAYER_H

#include <armadillo>
#include <QString>
#include <QStringList>

class Layer
{
protected:
    QString m_Name;

public:
    Layer(const QString &name);

    virtual ~Layer();

    virtual void Forward(arma::cube &input, arma::vec &output) = 0;
    virtual void Initialize_Weights(const QStringList &text) = 0;
    virtual void Initialize_Biases(const QStringList &text) = 0;

    QString Get_Name();
};

#endif // LAYER_H
