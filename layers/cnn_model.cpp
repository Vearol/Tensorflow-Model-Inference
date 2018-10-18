#include "cnn_model.h"

#include <QFile>
#include <QTextStream>
#include <QDebug>

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

    qInfo() << "Adding layer: " << name;

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

    auto all_Text = in.readAll();
    qInfo() << "File opened. Length: " << all_Text.size();

    auto layer_info_array = all_Text.split("New layer ");

    auto number_of_layers = layer_info_array.size();
    qInfo() << number_of_layers + 1 << " layers.";

    for (auto i = 0; i < number_of_layers; i++)
    {
        qInfo() << i;

        auto lines = layer_info_array[i].split('\n');

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
