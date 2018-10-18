#include "dense_layer.h"
#include <QDebug>

Dense_Layer::Dense_Layer(const QString &name)
    : Layer(name)
{
}

void Dense_Layer::Forward(arma::vec& input, arma::vec& output)
{
    output = (m_Weights * input) + m_Biases;
}

void Dense_Layer::Initialize_Weights(const QStringList &text)
{
    qInfo() << m_Name << " Initialize Weights";

    auto line_counter = 0;
    auto layer_parameters_text = text[line_counter].split(' ')[1].remove(0, 1).split(", ");

    auto input_flatten_size = layer_parameters_text[0].toInt();
    auto output_size = layer_parameters_text[1].toInt();

    m_Weights(input_flatten_size, output_size);

    for (auto i = 0; i < output_size; i++)
    {
        line_counter++;
        auto weights_text_line = text[line_counter].split(' ');

        for (auto j = 0; j < input_flatten_size; j++)
        {
            m_Weights.at(i, j) = weights_text_line[j].toFloat();
        }
    }
}

void Dense_Layer::Initialize_Biases(const QStringList &text)
{
    qInfo() << m_Name << " Initialize Biases";

    auto line_counter = 0;
    auto layer_parameters_text = text[line_counter].split(' ')[1].remove(0, 1).split(", ");

    auto bias_size = layer_parameters_text[0].toInt();
    m_Biases(bias_size);

    for (auto i = 1; i < bias_size; i++)
    {
        m_Biases.at(i) = text[i].toFloat();
    }
}
