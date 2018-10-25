#include "layer.h"

#include <QDebug>
#include <QDir>

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

void Layer::Initialize_Weights_and_Biases(const QString &layer_directory_path)
{
    QDir directory(layer_directory_path);
    QStringList npz_arrays = directory.entryList(QStringList() << "*.npz");

    foreach (auto array_path, npz_arrays)
    {
        if (array_path.contains("kernel"))
        {
            Initialize_Weights(array_path);
            continue;
        }

        if (array_path.contains("bias"))
        {
            Initialize_Biases(array_path);
        }
    }
}

void Layer::Initialize_Weights(const QString &array_path)
{
    qInfo() << "Layer " << m_Name << ". No Weights to initialize";
}

void Layer::Initialize_Biases(const QString &array_path)
{
    qInfo() << "Layer " << m_Name << ". No Biases to initialize";
}

QString Layer::Get_Name()
{
    return m_Name;
}
