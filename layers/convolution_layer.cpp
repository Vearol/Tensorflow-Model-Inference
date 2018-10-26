#include "convolution_layer.h"
#include "cnpy.h"

#include <QDebug>

Convolution_Layer::Convolution_Layer(const QString &name, short inputHeight, short inputWidth) : Physical_Layer(name)
{
    m_InputHeight = inputHeight;
    m_InputWidth = inputWidth;
    m_HorizontalStride = 1;
    m_VerticalStride = 1;
}

void Convolution_Layer::Forward(arma::cube& input, arma::cube& output)
{
    output = arma::zeros(m_InputHeight, m_InputWidth, m_NumFilters);

    for (auto filter = 0; filter < m_NumFilters; filter++)
        for (auto i = 0; i <= m_InputHeight - m_FilterHeight; i += m_VerticalStride)
            for (auto j = 0; j <= m_InputWidth - m_FilterWidth; j += m_HorizontalStride)
                output((i / m_VerticalStride), (j / m_HorizontalStride), filter) = arma::dot(
                            arma::vectorise(input.subcube(i, j, 0, i + m_FilterHeight - 1,
                                                          j + m_FilterWidth - 1, m_InputDepth - 1)),
                            arma::vectorise(m_Filters[filter]));

}

void Convolution_Layer::Initialize_Weights(const std::string &array_path)
{
    auto array = cnpy::npz_load(array_path, "arr_0");
    auto shape = array.shape;
    auto array_numbers = array.data<float>();

    m_FilterWidth = shape[0];
    m_FilterHeight = shape[1];
    m_InputDepth = shape[2];
    m_NumFilters = shape[3];

    m_Filters.reserve(m_NumFilters);

    for (auto filter = 0; filter < m_NumFilters; filter++)
    {
        m_Filters.push_back(arma::cube(m_FilterHeight, m_FilterWidth, m_InputDepth));
    }


    for (auto height = 0; height < m_FilterHeight; height++)
    {
        for (auto width = 0; width < m_FilterWidth; width++)
        {
            for (auto depth = 0; depth < m_InputDepth; depth++)
            {
                for (auto filter = 0; filter < m_NumFilters; filter++)
                {
                    auto index = height + width + depth + filter;
                    m_Filters[filter].at(height, width, depth) = array_numbers[index];
                }
            }
        }
    }
}
