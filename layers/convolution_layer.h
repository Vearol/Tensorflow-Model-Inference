#ifndef CONVOLUTION_LAYER_H
#define CONVOLUTION_LAYER_H

#include <QVector>
#include "physical_layer.h"

class Convolution_Layer : public Physical_Layer
{
private:
    short m_InputHeight;
    short m_InputWidth;
    short m_InputDepth;
    short m_FilterHeight;
    short m_FilterWidth;
    short m_HorizontalStride;
    short m_VerticalStride;
    short m_NumFilters;

    QVector<arma::cube> m_Filters;

    void Initialize_Weights(const QString &array_path);

public:
    Convolution_Layer(const QString &name,
            short inputHeight,
            short inputWidth);

    void Forward(arma::cube &input, arma::cube &output) override;    
};

#endif // CONVOLUTION_LAYER_H
