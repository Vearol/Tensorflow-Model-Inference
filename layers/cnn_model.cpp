#include "cnn_model.h"

#include <QDir>
#include <QDebug>

CNN_Model::CNN_Model()
{    
}

CNN_Model::CNN_Model(const QString &layers_directory_path)
{
    m_LayersDirectoryPath = layers_directory_path;
}

void CNN_Model::Add_Layer(Layer *layer)
{
    auto name = layer->Get_Name();

    if (m_Layers.contains(name)) return;

    m_Layers.insert(name, layer);
}

void CNN_Model::Load_Numbers_From_File()
{
    QDir layers_directory(m_LayersDirectoryPath);
    auto directories = layers_directory.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot);

    foreach (auto directory, directories)
    {
        auto layer_directory_path = directory.absoluteFilePath();

        auto layer_name = directory.absoluteFilePath().remove(m_LayersDirectoryPath).remove('/');

        if (m_Layers.contains(layer_name))
        {
            qInfo() << "Loading: " << layer_name;
            m_Layers[layer_name]->Initialize_Weights_and_Biases(layer_directory_path);
        }
    }
}
