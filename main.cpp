#include <QCoreApplication>
#include <QDebug>
#include <QMap>
#include <QFile>
#include <QImage>

#include "layers/layer.h"
#include "layers/cnn_model.h"
#include "layers/convolution_layer.h"
#include "layers/relu_layer.h"
#include "layers/max_pooling_layer.h"
#include "layers/dense_layer.h"
#include "layers/softmax_layer.h"

#include "cnpy.h"

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    QString path = "/home/lyubomyr/Projects/tiny_imagenet/multiple_nn/src/modelExport/saved_Layers";
    CNN_Model model(path);

    //56 x 56 x 3
    Layer* conv1_1 = new Convolution_Layer("conv1_1", 56, 56);
    Layer* relu1_1 = new ReLU_Layer("relu1_1");

    Layer* conv1_2 = new Convolution_Layer("conv1_2", 56, 56);
    Layer* relu1_2 = new ReLU_Layer("relu1_2");

    Layer* max_pool_1 = new Max_Pooling_Layer("max_pool_1", 56, 56, 64, 2, 2, 2, 2);

    //28 x 28 x 64
    Layer* conv2_1 = new Convolution_Layer("conv2_1", 28, 28);
    Layer* relu2_1 = new ReLU_Layer("relu2_1");

    Layer* conv2_2 = new Convolution_Layer("conv2_2", 28, 28);
    Layer* relu2_2 = new ReLU_Layer("relu2_2");

    Layer* max_pool_2 = new Max_Pooling_Layer("max_pool_2", 28, 28, 128, 2, 2, 2, 2);
    
    //14 x 14 x 128
    Layer* conv3_1 = new Convolution_Layer("conv3_1", 14, 14);
    Layer* relu3_1 = new ReLU_Layer("relu3_1");

    Layer* conv3_2 = new Convolution_Layer("conv3_2", 14, 14);
    Layer* relu3_2 = new ReLU_Layer("relu3_2");

    Layer* conv3_3 = new Convolution_Layer("conv3_3", 14, 14);
    Layer* relu3_3 = new ReLU_Layer("relu3_2");
    
    Layer* max_pool_3 = new Max_Pooling_Layer("max_pool_3", 14, 14, 256, 2, 2, 2, 2);
    
    //7 x 7 x 256
    Layer* conv4_1 = new Convolution_Layer("conv4_1", 7, 7);
    Layer* relu4_1 = new ReLU_Layer("relu4_1");

    Layer* conv4_2 = new Convolution_Layer("conv4_2", 7, 7);
    Layer* relu4_2 = new ReLU_Layer("relu4_2");

    Layer* conv4_3 = new Convolution_Layer("conv4_3", 7, 7);
    Layer* relu4_3 = new ReLU_Layer("relu4_2");
    
    Layer* max_pool_4 = new Max_Pooling_Layer("max_pool_4", 7, 7, 512, 2, 2, 2, 2);

    //7 x 7 x 512 -> (25088; 4096)
    Layer* dense_1 = new Dense_Layer("fc1");
    Layer* relu_dense1 = new ReLU_Layer("relu_dense1");

    //4096 x 2048
    Layer* dense_2 = new Dense_Layer("fc2");
    Layer* relu_dense2 = new ReLU_Layer("relu_dense2");

    //2048 x 200
    Layer* dense_3 = new Dense_Layer("fc3");

    Layer* soft_max = new Softmax_Layer("soft_max");

    model.Add_Layer(conv1_1);
    model.Add_Layer(conv1_2);
    model.Add_Layer(conv2_1);
    model.Add_Layer(conv2_2);
    model.Add_Layer(conv3_1);
    model.Add_Layer(conv3_2);
    model.Add_Layer(conv3_3);
    model.Add_Layer(conv4_1);
    model.Add_Layer(conv4_2);
    model.Add_Layer(conv4_3);
    model.Add_Layer(relu1_1);
    model.Add_Layer(relu1_2);
    model.Add_Layer(relu2_1);
    model.Add_Layer(relu2_2);
    model.Add_Layer(relu3_1);
    model.Add_Layer(relu3_2);
    model.Add_Layer(relu3_3);
    model.Add_Layer(relu4_1);
    model.Add_Layer(relu4_2);
    model.Add_Layer(relu4_3);
    model.Add_Layer(relu_dense1);
    model.Add_Layer(relu_dense2);
    model.Add_Layer(max_pool_1);
    model.Add_Layer(max_pool_2);
    model.Add_Layer(max_pool_3);
    model.Add_Layer(max_pool_4);
    model.Add_Layer(dense_1);
    model.Add_Layer(dense_2);
    model.Add_Layer(dense_3);
    model.Add_Layer(soft_max);

    model.Load_Numbers_From_File();

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

    QImage image;
    image.load("/home/lyubomyr/Projects/tiny_imagenet/multiple_nn/tiny-imagenet-200/test/images/test_4.JPEG");

    arma::cube input = arma::zeros(56, 56, 3);

    for (auto i = 0; i < 56; i++)
    {
        for (auto j = 0; j < 56; j++)
        {
            auto pixel = image.pixel(i, j);

            input.at(i, j, 0) = qRed(pixel) / 255.f;
            input.at(i, j, 1) = qGreen(pixel) / 255.f;
            input.at(i, j, 2) = qBlue(pixel) / 255.f;
        }
    }

    conv1_1->Forward(input, conv1_1_out);
    relu1_1->Forward(conv1_1_out, relu1_1_out);
    conv1_2->Forward(relu1_1_out, conv1_2_out);
    relu1_2->Forward(conv1_1_out, relu1_2_out);
    max_pool_1->Forward(relu1_2_out, max_pool_1_out);
    conv2_1->Forward(max_pool_1_out, conv2_1_out);
    relu2_1->Forward(conv2_1_out, relu2_1_out);
    conv2_2->Forward(relu2_1_out, conv2_2_out);
    relu2_2->Forward(conv2_1_out, relu2_2_out);
    max_pool_2->Forward(relu2_2_out, max_pool_2_out);
    conv3_1->Forward(max_pool_2_out, conv3_1_out);
    relu3_1->Forward(conv3_1_out, relu3_1_out);
    conv3_2->Forward(relu3_1_out, conv3_2_out);
    relu3_2->Forward(conv3_1_out, relu3_2_out);
    conv3_3->Forward(relu3_2_out, conv3_3_out);
    relu3_3->Forward(conv3_3_out, relu3_3_out);
    max_pool_3->Forward(relu3_3_out, max_pool_3_out);
    conv4_1->Forward(max_pool_3_out, conv4_1_out);
    relu4_1->Forward(conv4_1_out, relu4_1_out);
    conv4_2->Forward(relu4_1_out, conv4_2_out);
    relu4_2->Forward(conv4_1_out, relu4_2_out);
    conv4_3->Forward(relu4_2_out, conv4_3_out);
    relu4_3->Forward(conv4_3_out, relu4_3_out);
    arma::vec flatten = arma::vectorise(relu4_3_out);
    dense_1->Forward(flatten, dense_1_out);
    relu_dense1->Forward(dense_1_out, relu_dense1_out);
    dense_2->Forward(relu_dense1_out, dense_2_out);
    relu_dense2->Forward(dense_2_out, relu_dense2_out);
    dense_3->Forward(relu_dense2_out, dense_3_out);

    QMap<float, QString> output_label;
    QMap<QString, QString> label_text;

    QFile outPutLabesFile("/home/lyubomyr/Projects/tiny_imagenet/multiple_nn/tiny-imagenet-200/wnids.txt");
    if (outPutLabesFile.open(QIODevice::ReadOnly))
    {
        auto i = 0;
        QTextStream in(&outPutLabesFile);
        while (!in.atEnd())
        {
            auto line = in.readLine();

            output_label.insert(dense_3_out.at(i), line);
        }
        outPutLabesFile.close();
    }

    QFile labelsTextFile("/home/lyubomyr/Projects/tiny_imagenet/multiple_nn/tiny-imagenet-200/words.txt");
    if (labelsTextFile.open(QIODevice::ReadOnly))
    {
        QTextStream in(&labelsTextFile);

        auto output_labes = output_label.values();
        while (!in.atEnd())
        {
            auto line = in.readLine();
            auto name_labes = line.split('\t');

            if (output_labes.contains(name_labes[0]))
            {
                label_text.insert(name_labes[0], name_labes[1]);
            }
        }
        labelsTextFile.close();
    }

    auto top_N = 5;
    QSet<QString> top_N_labels;
    top_N_labels.reserve(top_N);

    for (auto top = 0; top < top_N; top++)
    {
        auto current_max = -1;

        for (auto i = 0; i < 200; i++)
        {
            auto current_prediction = dense_3_out.at(i);

            if (current_prediction > current_max)
            {
                if (top_N_labels.contains(output_label[current_prediction])) continue;

                current_max = current_prediction;
            }
        }

        auto current_max_label = output_label[current_max];

        top_N_labels.insert(current_max_label);
        qInfo() << label_text[current_max_label];
    }

    return a.exec();
}
