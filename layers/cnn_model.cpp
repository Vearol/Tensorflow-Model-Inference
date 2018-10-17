#include "cnn_model.h"
#include <QFile>
#include <QTextStream>

CNN_Model::CNN_Model()
{    
}

CNN_Model::CNN_Model(const QString &path)
{
    m_Model_File_Path = path;
}

void CNN_Model::Add_Layer(Layer *layer)
{
    auto name = layer->Get_Name();

    if (m_Layers.contains(name)) return;

    m_Layers.insert(name, layer);
}

void CNN_Model::Load_Numbers_From_File()
{
    QFile file(m_Model_File_Path);

    if(!file.open(QIODevice::ReadOnly)) {
        return;
    }

    QTextStream in(&file);

    auto line = in.readAll();

    auto layer_info_array = line.split("New layer ");
    for (auto layer_info : layer_info_array)
    {
        auto lines = layer_info.split('\n');
        auto file_name_line = lines[0];
        auto file_name = file_name_line.split('/')[0];

        if (file_name_line.contains("bias"))
        {
            m_Layers[file_name]->Initialize_Weights(lines);
            continue;
        }

        m_Layers[file_name]->Initialize_Weights(lines);
    }

    file.close();
}
