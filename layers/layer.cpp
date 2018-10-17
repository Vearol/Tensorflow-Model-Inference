#include "layer.h"

Layer::Layer(const QString &name)
{
    m_Name = name;
}

Layer::~Layer()
{
}

QString Layer::Get_Name()
{
    return m_Name;
}
