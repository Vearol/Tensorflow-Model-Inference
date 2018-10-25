#include "cnn_model.h"

#include <QDirIterator>
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
    QDirIterator dir_iterator(m_LayersDirectoryPath, QDir::Dirs | QDir::NoSymLinks | QDir::NoDotAndDotDot, QDirIterator::Subdirectories);

    while (dir_iterator.hasNext())
    {
        auto current_directory = dir_iterator.next();
        auto layer_directory_path = current_directory;
        auto layer_name = current_directory.remove(m_LayersDirectoryPath).remove('/');

        if (m_Layers.contains(layer_name))
        {
            m_Layers[layer_name]->Initialize_Weights_and_Biases(layer_directory_path);
        }
    }
}
