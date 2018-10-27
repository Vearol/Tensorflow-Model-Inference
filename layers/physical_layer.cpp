#include "physical_layer.h"
#include "cnpy.h"

#include <QDir>
#include <QDebug>

Physical_Layer::Physical_Layer(const QString &name) : Layer(name)
{

}

void Physical_Layer::initialize_weights_and_Biases(const QString &layer_directory_path)
{
    QDir directory(layer_directory_path);
    auto npz_arrays = directory.entryInfoList(QStringList() << "*.npz");

    foreach (auto array_path, npz_arrays)
    {
        auto full_path = array_path.absoluteFilePath();

        if (full_path.contains("kernel"))
        {
            initialize_weights(full_path.toStdString());
            continue;
        }

        if (full_path.contains("bias"))
        {
            initialize_biases(full_path.toStdString());
        }
    }
}

void Physical_Layer::initialize_biases(const std::string &array_path)
{
    auto array = cnpy::npz_load(array_path, "arr_0");
    auto shape = array.shape[0];
    auto array_numbers = array.data<float>();

    m_Biases = arma::zeros(shape);

    for (auto i = 0; i < shape; i++)
    {
        m_Biases.at(i) = array_numbers[i];
    }
}
