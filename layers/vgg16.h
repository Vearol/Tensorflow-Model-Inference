#ifndef VGG16_H
#define VGG16_H

#include "cnn_model.h"
#include "layers/convolution_layer.h"
#include "layers/relu_layer.h"
#include "layers/max_pooling_layer.h"
#include "layers/dense_layer.h"
#include "layers/softmax_layer.h"

class VGG16 : public CNN_Model
{
public:
    VGG16(const QString &layers_directory_path);

    void init();

    void forward(const QString &image_path);
    void top_n(int n);

private:
    arma::cube image_to_cube(const QString &image_path);

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
};

#endif // VGG16_H
