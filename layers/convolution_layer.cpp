#include "convolution_layer.h"

Convolution_Layer::Convolution_Layer(const QString &name, short inputHeight, short inputWidth, short horizontalStride,
                                     short verticalStride) : Layer(name)
{
    m_InputHeight = inputHeight;
    m_InputWidth = inputWidth;
    m_HorizontalStride = horizontalStride;
    m_VerticalStride = verticalStride;
}

void Convolution_Layer::Forward(arma::cube& input, arma::cube& output)
{
    output = arma::zeros((m_InputHeight - m_FilterHeight) / m_VerticalStride + 1,
                         (m_InputWidth - m_FilterWidth) / m_HorizontalStride + 1,
                         m_NumFilters);

    for (auto filter = 0; filter < m_NumFilters; filter++)
        for (auto i = 0; i <= m_InputHeight - m_FilterHeight; i += m_VerticalStride)
            for (auto j = 0; j <= m_InputWidth - m_FilterWidth; j += m_HorizontalStride)
                output((i / m_VerticalStride), (j / m_HorizontalStride), filter) = arma::dot(
                            arma::vectorise(
                                input.subcube(i, j, 0,
                                              i + m_FilterHeight - 1, j + m_FilterWidth - 1, m_InputDepth - 1)
                                ),
                            arma::vectorise(m_Filters[filter]));

}

void Convolution_Layer::Initialize_Weights(const QStringList &text)
{
    auto line_counter = 0;
    auto layer_parameters_text = text[line_counter].split(' ')[1].remove(0, 1).split(", ");

    m_FilterHeight = layer_parameters_text[0].toInt();
    m_FilterWidth = layer_parameters_text[1].toInt();
    m_InputDepth = layer_parameters_text[2].toInt();
    m_NumFilters = layer_parameters_text[3].toInt();

    m_Filters.reserve(m_NumFilters);

    for (auto filter = 0; filter < m_NumFilters; filter++)
    {
        m_Filters.push_back(arma::cube(m_FilterHeight, m_FilterWidth, m_InputDepth));
    }

    for (auto height = 0; height < m_FilterHeight; height++)
    {
        line_counter++;

        for (auto width = 0; width < m_FilterWidth; width++)
        {
            line_counter++;

            for (auto depth = 0; depth < m_InputDepth; depth++)
            {
                line_counter++;
                auto weights_text_line = text[line_counter].split(' ');

                for (auto filter = 0; filter < m_NumFilters; filter++)
                {
                    m_Filters[filter](height, width, depth) = weights_text_line[filter].toFloat();
                }
            }
        }
    }
}

void Convolution_Layer::Initialize_Biases(const QStringList &text)
{
    m_Biases.resize(m_NumFilters);

    for (auto i = 1; i < m_NumFilters; i++)
    {
        m_Biases.push_back(text[i].toFloat());
    }
}
