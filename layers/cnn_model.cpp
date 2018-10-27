#include "cnn_model.h"

#include <QDir>
#include <QDebug>
#include <QFile>

CNN_Model::CNN_Model(const QString &layers_directory_path)
{
    m_Layers_directory_path = layers_directory_path;
}

void CNN_Model::add_layer(Layer *layer)
{
    auto name = layer->get_name();

    if (m_Layers.contains(name)) return;

    m_Layers.insert(name, layer);
}

void CNN_Model::load_numbers_from_file()
{
    QDir layers_directory(m_Layers_directory_path);
    auto directories = layers_directory.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot);

    foreach (auto directory, directories)
    {
        auto layer_directory_path = directory.absoluteFilePath();

        auto layer_name = directory.absoluteFilePath().remove(m_Layers_directory_path).remove('/');

        if (m_Layers.contains(layer_name))
        {
            qInfo() << "Loading: " << layer_name;
            m_Layers[layer_name]->initialize_weights_and_Biases(layer_directory_path);
        }
    }
}

void CNN_Model::get_predicted_labels()
{
    QFile output_labes_file("/home/lyubomyr/Projects/tiny_imagenet/multiple_nn/tiny-imagenet-200/wnids.txt");
    if (output_labes_file.open(QIODevice::ReadOnly))
    {
        auto i = 0;
        QTextStream in(&output_labes_file);
        while (!in.atEnd())
        {
            auto line = in.readLine();

            m_Output_labels.insert(m_Model_output.at(i), line);
        }
        output_labes_file.close();
    }

    QFile labels_text_file("/home/lyubomyr/Projects/tiny_imagenet/multiple_nn/tiny-imagenet-200/words.txt");
    if (labels_text_file.open(QIODevice::ReadOnly))
    {
        QTextStream in(&labels_text_file);

        auto output_labes = m_Output_labels.values();
        while (!in.atEnd())
        {
            auto line = in.readLine();
            auto name_labes = line.split('\t');

            if (output_labes.contains(name_labes[0]))
            {
                m_Labels_text.insert(name_labes[0], name_labes[1]);
            }
        }
        labels_text_file.close();
    }

    qInfo() << "Loaded labels: " << m_Labels_text.size() << m_Output_labels.size();
}
