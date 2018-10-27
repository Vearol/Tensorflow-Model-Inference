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

void Layer::forward(arma::cube &input, arma::cube &output) { }

void Layer::forward(arma::vec &input, arma::vec &output) { }

void Layer::initialize_weights_and_Biases(const QString &layer_directory_path) { }

QString Layer::get_name()
{
    return m_Name;
}
