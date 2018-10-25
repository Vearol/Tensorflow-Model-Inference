#include <QCoreApplication>
#include <QDebug>

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

    //cnpy::NpyArray arr2 = cnpy::npz_load("/home/lyubomyr/Projects/tiny_imagenet/multiple_nn/src/modelExport/saved_Layers/conv1_1/kernel:0.npz","arr_0");

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

    Layer* conv3_3 = new Convolution_Layer("conv3_2", 14, 14);
    Layer* relu3_3 = new ReLU_Layer("relu3_2");
    
    Layer* max_pool_3 = new Max_Pooling_Layer("max_pool_3", 14, 14, 256, 2, 2, 2, 2);
    
    //7 x 7 x 256
    Layer* conv4_1 = new Convolution_Layer("conv4_1", 7, 7);
    Layer* relu4_1 = new ReLU_Layer("relu4_1");

    Layer* conv4_2 = new Convolution_Layer("conv4_2", 7, 7);
    Layer* relu4_2 = new ReLU_Layer("relu4_2");

    Layer* conv4_3 = new Convolution_Layer("conv4_2", 7, 7);
    Layer* relu4_3 = new ReLU_Layer("relu4_2");
    
    Layer* max_pool_4 = new Max_Pooling_Layer("max_pool_4", 7, 7, 512, 2, 2, 2, 2);

    //7 x 7 x 512 -> (25088; 4096)
    Layer* dense1 = new Dense_Layer("fc1");

    //4096 x 2048
    Layer* dense2 = new Dense_Layer("fc2");

    //2048 x 200
    Layer* dense3 = new Dense_Layer("fc3");

    Layer* soft_max = new Softmax_Layer("soft_max");

    qInfo() << "Created Layers";

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
    model.Add_Layer(max_pool_1);
    model.Add_Layer(max_pool_2);
    model.Add_Layer(max_pool_3);
    model.Add_Layer(max_pool_4);
    model.Add_Layer(dense1);
    model.Add_Layer(dense2);
    model.Add_Layer(dense3);
    model.Add_Layer(soft_max);

    model.Load_Numbers_From_File();
    */

    return a.exec();
}
