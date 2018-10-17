#ifndef CNN_MODEL_H
#define CNN_MODEL_H

#include <QMap>

#include "layer.h"

class CNN_Model
{
public:
    CNN_Model();
    CNN_Model(const QString &path);

    void Add_Layer(Layer* layer);
    void Load_Numbers_From_File();

private:
    QString m_Model_File_Path;
    QMap<QString, Layer*> m_Layers;
};

#endif // CNN_MODEL_H
