#include "dense_layer.h"
#include "cnpy.h"

#include <QDebug>

Dense_Layer::Dense_Layer(const QString &name): Physical_Layer(name)
{
}

void Dense_Layer::Forward(arma::vec& input, arma::vec& output)
{
    output = (m_Weights * input) + m_Biases;
}

void Dense_Layer::Initialize_Weights(const std::string &array_path)
{
    auto array = cnpy::npz_load(array_path, "arr_0");
    auto shape = array.shape;
    auto array_numbers = array.data<float>();

    auto input_flatten_size = shape[0];
    auto output_size = shape[1];

    m_Weights = arma::zeros(output_size, input_flatten_size);

    auto index = 0;
    for (auto i = 0; i < output_size; i++)
    {
        for (auto j = 0; j < input_flatten_size; j++)
        {
            index++;
            m_Weights.at(i, j) = array_numbers[index];
        }
    }

}
