#include "layer.h"

#include <QDebug>

Layer::Layer(const QString &name)
{
    m_Name = name;
}

Layer::~Layer()
{
}

void Layer::Forward(arma::cube &input, arma::cube &output)
{
}

void Layer::Forward(arma::vec &input, arma::vec &output)
{

}

void Layer::Initialize_Weights(const QStringList &text)
{
    qInfo() << "Layer " << m_Name << ". No Weights to initialize";
}

void Layer::Initialize_Biases(const QStringList &text)
{
    qInfo() << "Layer " << m_Name << ". No Biases to initialize";
}

QString Layer::Get_Name()
{
    return m_Name;
}
