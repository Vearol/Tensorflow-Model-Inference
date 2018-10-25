#include "physical_layer.h"
#include "cnpy.h"

#include <QDir>
#include <QDebug>

Physical_Layer::Physical_Layer(const QString &name) : Layer(name)
{

}

void Physical_Layer::Initialize_Weights_and_Biases(const QString &layer_directory_path)
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

void Physical_Layer::Initialize_Biases(const QString &array_path)
{
    qInfo() << m_Name << array_path;

    auto array = cnpy::npz_load(array_path.toStdString(), "arr_0");
    auto shape = array.shape[0];
    auto array_numbers = array.data<float>();

    m_Biases = arma::zeros(shape);

    for (auto i = 0; i < shape; i++)
    {
        m_Biases.at(i) = array_numbers[i];
    }
}
