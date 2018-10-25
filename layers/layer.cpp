#include "layer.h"
#include "cnpy.h"

#include <QDebug>
#include <QDir>

Layer::Layer(const QString &name)
{
    m_Name = name;
}

Layer::~Layer()
{
}

void Layer::Forward(arma::cube &input, arma::cube &output) { }

void Layer::Forward(arma::vec &input, arma::vec &output) { }

void Layer::Initialize_Weights_and_Biases(const QString &layer_directory_path) { }

QString Layer::Get_Name()
{
    return m_Name;
}
