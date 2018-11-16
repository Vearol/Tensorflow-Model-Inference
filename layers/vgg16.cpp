#include "vgg16.h"

#include <QImage>
#include <QFile>
#include <QDebug>
#include <QSet>
#include <QDebug>

VGG16::VGG16(const QString &layers_directory_path) : CNN_Model(layers_directory_path)
{
}

void VGG16::init()
{
    add_layer(conv1_1);
    add_layer(conv1_2);
    add_layer(conv2_1);
    add_layer(conv2_2);
    add_layer(conv3_1);
    add_layer(conv3_2);
    add_layer(conv3_3);
    add_layer(conv4_1);
    add_layer(conv4_2);
    add_layer(conv4_3);
    add_layer(relu1_1);
    add_layer(relu1_2);
    add_layer(relu2_1);
    add_layer(relu2_2);
    add_layer(relu3_1);
    add_layer(relu3_2);
    add_layer(relu3_3);
    add_layer(relu4_1);
    add_layer(relu4_2);
    add_layer(relu4_3);
    add_layer(relu_dense1);
    add_layer(relu_dense2);
    add_layer(max_pool_1);
    add_layer(max_pool_2);
    add_layer(max_pool_3);
    add_layer(max_pool_4);
    add_layer(dense_1);
    add_layer(dense_2);
    add_layer(dense_3);
    add_layer(soft_max);
}

void VGG16::forward(const QString &image_path)
{
    arma::cube conv1_1_out = arma::zeros(56, 56, 64);
    arma::cube conv1_2_out = arma::zeros(56, 56, 64);
    arma::cube conv2_1_out = arma::zeros(28, 28, 128);
    arma::cube conv2_2_out = arma::zeros(28, 28, 128);
    arma::cube conv3_1_out = arma::zeros(14, 14, 256);
    arma::cube conv3_2_out = arma::zeros(14, 14, 256);
    arma::cube conv3_3_out = arma::zeros(14, 14, 256);
    arma::cube conv4_1_out = arma::zeros(7, 7, 512);
    arma::cube conv4_2_out = arma::zeros(7, 7, 512);
    arma::cube conv4_3_out = arma::zeros(7, 7, 512);
    arma::cube relu1_1_out = arma::zeros(56, 56, 64);
    arma::cube relu1_2_out = arma::zeros(56, 56, 64);
    arma::cube relu2_1_out = arma::zeros(14, 14, 256);
    arma::cube relu2_2_out = arma::zeros(14, 14, 256);
    arma::cube relu3_1_out = arma::zeros(14, 14, 256);
    arma::cube relu3_2_out = arma::zeros(14, 14, 256);
    arma::cube relu3_3_out = arma::zeros(14, 14, 256);
    arma::cube relu4_1_out = arma::zeros(7, 7, 512);
    arma::cube relu4_2_out = arma::zeros(7, 7, 512);
    arma::cube relu4_3_out = arma::zeros(7, 7, 512);
    arma::vec relu_dense1_out = arma::zeros(4096);
    arma::vec relu_dense2_out = arma::zeros(2048);
    arma::cube max_pool_1_out = arma::zeros(28, 28, 128);
    arma::cube max_pool_2_out = arma::zeros(14, 14, 256);
    arma::cube max_pool_3_out = arma::zeros(7, 7, 512);
    arma::vec dense_1_out = arma::zeros(4096);
    arma::vec dense_2_out = arma::zeros(2048);
    arma::vec dense_3_out = arma::zeros(200);
    arma::vec softmax_out = arma::zeros(200);

    auto input = image_to_cube(image_path);

    conv1_1->forward(input, conv1_1_out);
    relu1_1->forward(conv1_1_out, relu1_1_out);
    conv1_2->forward(relu1_1_out, conv1_2_out);
    relu1_2->forward(conv1_1_out, relu1_2_out);
    max_pool_1->forward(relu1_2_out, max_pool_1_out);
    conv2_1->forward(max_pool_1_out, conv2_1_out);
    relu2_1->forward(conv2_1_out, relu2_1_out);
    conv2_2->forward(relu2_1_out, conv2_2_out);
    relu2_2->forward(conv2_1_out, relu2_2_out);
    max_pool_2->forward(relu2_2_out, max_pool_2_out);
    conv3_1->forward(max_pool_2_out, conv3_1_out);
    relu3_1->forward(conv3_1_out, relu3_1_out);
    conv3_2->forward(relu3_1_out, conv3_2_out);
    relu3_2->forward(conv3_1_out, relu3_2_out);
    conv3_3->forward(relu3_2_out, conv3_3_out);
    relu3_3->forward(conv3_3_out, relu3_3_out);
    max_pool_3->forward(relu3_3_out, max_pool_3_out);
    conv4_1->forward(max_pool_3_out, conv4_1_out);
    relu4_1->forward(conv4_1_out, relu4_1_out);
    conv4_2->forward(relu4_1_out, conv4_2_out);
    relu4_2->forward(conv4_1_out, relu4_2_out);
    conv4_3->forward(relu4_2_out, conv4_3_out);
    relu4_3->forward(conv4_3_out, relu4_3_out);
    arma::vec flatten = arma::vectorise(relu4_3_out);
    dense_1->forward(flatten, dense_1_out);
    relu_dense1->forward(dense_1_out, relu_dense1_out);
    dense_2->forward(relu_dense1_out, dense_2_out);
    relu_dense2->forward(dense_2_out, relu_dense2_out);
    dense_3->forward(relu_dense2_out, dense_3_out);

    m_Model_output = dense_3_out;

    auto max_num = -1.f;
    auto max_index = -1;
    for (auto i = 0; i < 200; i++){
        if (m_Model_output.at(i) > max_num){
            max_num = m_Model_output.at(i);
            max_index = i;
            qInfo() << m_Model_output.at(i);
        }
    }

    qInfo() << "Max: " << max_num << ", " << max_index;
}

void VGG16::top_n(int n)
{
    QMap<int, QString> top_N_labels;

    for (auto top = 0; top < n; top++)
    {
        auto current_max_prediction = -1.f;
        auto current_max_index = 0;

        for (auto i = 0; i < 200; i++)
        {
            auto current_prediction = m_Model_output.at(i);

            if (current_prediction > current_max_prediction)
            {
                if (top_N_labels.contains(i)) continue;

                current_max_prediction = current_prediction;
                current_max_index = i;
            }
        }

        auto current_max_label = m_Output_labels[current_max_index];

        top_N_labels.insert(current_max_index, current_max_label);
        qInfo() << m_Model_output[current_max_index] << ": " << m_Labels_text[current_max_label];
    }
}

arma::cube VGG16::image_to_cube(const QString &image_path)
{
    QImage image;
    image.load(image_path);

    arma::cube input = arma::zeros(56, 56, 3);

    for (auto i = 0; i < 56; i++)
    {
        for (auto j = 0; j < 56; j++)
        {
            auto pixel = image.pixel(i+4, j+4);

            input.at(i, j, 0) = qRed(pixel) / 255.f;
            input.at(i, j, 1) = qGreen(pixel) / 255.f;
            input.at(i, j, 2) = qBlue(pixel) / 255.f;
        }
    }

    return input;
}
