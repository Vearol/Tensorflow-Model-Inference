#ifndef MAX_POOLING_LAYER_H
#define MAX_POOLING_LAYER_H

#include "functional_layer.h"

class Max_Pooling_Layer : public Functional_layer
{
private:
    short m_InputHeight;
    short m_InputWidth;
    short m_InputDepth;
    short m_PoolingWindowHeight;
    short m_PoolingWindowWidth;
    short m_VerticalStride;
    short m_HorizontalStride;

public:
    Max_Pooling_Layer(
            const QString &name,
            short inputHeight,
            short inputWidth,
            short inputDepth,
            short poolingWindowHeight,
            short poolingWindowWidth,
            short verticalStride,
            short horizontalStride);

    void forward(arma::cube &input, arma::cube &output) override;
};

#endif // MAX_POOLING_LAYER_H
