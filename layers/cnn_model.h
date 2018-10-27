#ifndef CNN_MODEL_H
#define CNN_MODEL_H

#include <QMap>

#include "layer.h"

class CNN_Model
{
public:
    CNN_Model(const QString &layers_directory_path);

    void load_numbers_from_file();
    void get_predicted_labels();

    virtual void forward(const QString &image_path) = 0;
    virtual void init() = 0;
    virtual void top_n(int n) = 0;

protected:
    QString m_Layers_directory_path;
    QMap<QString, Layer*> m_Layers;

    QMap<float, QString> m_Output_labels;
    QMap<QString, QString> m_Labels_text;

    arma::vec m_Model_output = arma::zeros(200);

    void add_layer(Layer* layer);
};

#endif // CNN_MODEL_H
