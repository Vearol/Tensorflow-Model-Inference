#include "max_pooling_layer.h"

Max_Pooling_Layer::Max_Pooling_Layer(const QString &name, short inputHeight, short inputWidth, short inputDepth,
                                     short poolingWindowHeight, short poolingWindowWidth, short verticalStride,
                                     short horizontalStride) : Functional_layer(name)
{
    m_InputHeight = inputHeight;
    m_InputWidth = inputWidth;
    m_InputDepth = inputDepth;
    m_PoolingWindowHeight = poolingWindowHeight;
    m_PoolingWindowWidth = poolingWindowWidth;
    m_VerticalStride = verticalStride;
    m_HorizontalStride = horizontalStride;
}

void Max_Pooling_Layer::forward(arma::cube& input, arma::cube& output)
{
    output = arma::zeros(
                (m_InputHeight - m_PoolingWindowHeight) / m_VerticalStride + 1,
                (m_InputWidth - m_PoolingWindowWidth) / m_HorizontalStride + 1,
                m_InputDepth
                );
    for (auto slide = 0; slide < m_InputDepth; slide ++)
    {
        for (auto i = 0;
             i <= m_InputHeight - m_PoolingWindowHeight;
             i += m_VerticalStride)
        {
            for (auto j = 0;
                 j <= m_InputWidth - m_PoolingWindowWidth;
                 j += m_HorizontalStride)
            {
                output.slice(slide)( i / m_VerticalStride, j / m_HorizontalStride) =
                        input.slice(slide).submat(i,
                                                 j,
                                                 i + m_PoolingWindowHeight - 1,
                                                 j + m_PoolingWindowWidth - 1)
                        .max();
            }
        }
    }
}
