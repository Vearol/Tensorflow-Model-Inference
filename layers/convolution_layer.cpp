#include "convolution_layer.h"
#include "cnpy.h"
#include "cmath"

#include <QDebug>

Convolution_Layer::Convolution_Layer(const QString &name, short inputHeight, short inputWidth) : Physical_Layer(name)
{
    m_InputHeight = inputHeight;
    m_InputWidth = inputWidth;
    m_HorizontalStride = 1;
    m_VerticalStride = 1;
}

void Convolution_Layer::forward(arma::cube& input, arma::cube& output)
{
    output = arma::zeros(m_InputHeight, m_InputWidth, m_NumFilters);

    auto filter_radius_x = m_FilterWidth / 2;
    auto filter_radius_y = m_FilterHeight / 2;

    for (auto filter = 0; filter < m_NumFilters; filter++)
        for (auto i = 0; i < m_InputHeight; i += m_VerticalStride)
            for (auto j = 0; j < m_InputWidth; j += m_HorizontalStride)
            {
                auto input_x_top = j - filter_radius_x;
                auto input_y_top = i - filter_radius_y;

                auto input_x_bot = j + filter_radius_x;
                auto input_y_bot = i + filter_radius_y;

                auto filter_x_top = 0;
                auto filter_y_top = 0;

                auto filter_x_bot = m_FilterWidth - 1;
                auto filter_y_bot = m_FilterHeight - 1;

                if (input_x_top < 0)
                {
                    input_x_top = 0;
                    filter_x_top = filter_radius_x - j;;
                }

                if (input_y_top < 0)
                {
                    input_y_top = 0;
                    filter_y_top = filter_radius_y - i;;
                }

                if (input_x_bot > m_InputWidth - 1)
                {
                    input_x_bot = m_InputWidth - 1;
                    filter_x_bot = filter_radius_x + m_InputWidth - j - 1;;
                }

                if (input_y_bot > m_InputHeight - 1)
                {
                    input_y_bot = m_InputHeight - 1;
                    filter_y_bot = filter_radius_y + m_InputHeight - i - 1;;
                }

                auto neuron_value = arma::dot(
                            arma::vectorise(input.subcube(input_y_top, input_x_top, 0,
                                                          input_y_bot, input_x_bot, m_InputDepth - 1)),
                            arma::vectorise(m_Filters[filter].subcube(filter_y_top, filter_x_top, 0,
                                                                      filter_y_bot, filter_x_bot, m_InputDepth - 1)));

                output((i / m_VerticalStride), (j / m_HorizontalStride), filter) = neuron_value + m_Biases[filter];
            }
}

void Convolution_Layer::initialize_weights(const std::string &array_path)
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

    auto index = 0;
    for (auto height = 0; height < m_FilterHeight; height++)
    {
        for (auto width = 0; width < m_FilterWidth; width++)
        {
            for (auto depth = 0; depth < m_InputDepth; depth++)
            {
                for (auto filter = 0; filter < m_NumFilters; filter++)
                {
                    m_Filters[filter].at(height, width, depth) = array_numbers[index];

                    index++;
                }
            }
        }
    }
}
