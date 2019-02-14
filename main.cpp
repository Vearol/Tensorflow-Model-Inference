#include <algorithm>
#include <iostream>
#include <iomanip>
#include <vector>
#include <memory>
#include <utility>

#include <QCoreApplication>
#include <QDebug>
#include <QImage>

#include <cnpy/cnpy.h>

#include <layers/layer_base.h>
#include <layers/convolutionlayer.h>
#include <layers/fullyconnectedlayer.h>
#include <layers/poolinglayer.h>
#include <network/network2.h>
#include <network/activator.h>
#include <common/array3d.h>
#include <common/array3d_math.h>
#include <common/log.h>

#include "loaders/fs_loader_factory.h"
#include "parsing/parsed_labels.h"

#define STRINGIZE_(x) #x
#define STRINGIZE(x) STRINGIZE_(x)

const QString testsDir = STRINGIZE(TESTS_DIR);

using named_layers_list = std::vector<std::pair<std::shared_ptr<yannpp::layer_base_t<float>>, std::string>>;
using layers_list = std::vector<std::shared_ptr<yannpp::layer_base_t<float>>>;

yannpp::activator_t<float> relu_activator(yannpp::relu_v<float>, yannpp::relu_v<float>);

layers_list create_layers() {
    using namespace yannpp;

    using cnn_t = convolution_layer_t<float>;
    using fc_t = fully_connected_layer_t<float>;
    using pl_t = pooling_layer_t<float>;

    named_layers_list layers = {
        /* ############## CONVOLUTION LAYER 1 ################# */
        { std::make_shared<cnn_t>(
          shape3d_t(56, 56, 3), // input
          shape3d_t(3, 3, 3), // filter
          64, // filters count
          1, // stride
          padding_type::same,
          relu_activator), "conv1_1" },
        { std::make_shared<cnn_t>(
          shape3d_t(56, 56, 64), // input
          shape3d_t(3, 3, 64), // filter
          64, // filters count
          1, // stride
          padding_type::same,
          relu_activator), "conv1_2" },
        { std::make_shared<pl_t>(
          2, // window
          2 // stride
          ), "pool1" },
        /* ############## CONVOLUTION LAYER 2 ################# */
        { std::make_shared<cnn_t>(
          shape3d_t(28, 28, 64), // input
          shape3d_t(3, 3, 64), // filter
          128, // filters count
          1, // stride
          padding_type::same,
          relu_activator), "conv2_1" },
        { std::make_shared<cnn_t>(
          shape3d_t(28, 28, 128), // input
          shape3d_t(3, 3, 128), // filter
          128, // filters count
          1, // stride
          padding_type::same,
          relu_activator), "conv2_2" },
        { std::make_shared<pl_t>(
          2, // window
          2 // stride
          ), "pool2" },
        /* ############## CONVOLUTION LAYER 3 ################# */
        { std::make_shared<cnn_t>(
          shape3d_t(14, 14, 128), // input
          shape3d_t(3, 3, 128), // filter
          256, // filters count
          1, // stride
          padding_type::same,
          relu_activator), "conv3_1" },
        { std::make_shared<cnn_t>(
          shape3d_t(14, 14, 256), // input
          shape3d_t(3, 3, 256), // filter
          256, // filters count
          1, // stride
          padding_type::same,
          relu_activator), "conv3_2" },
        { std::make_shared<cnn_t>(
          shape3d_t(14, 14, 256), // input
          shape3d_t(3, 3, 256), // filter
          256, // filters count
          1, // stride
          padding_type::same,
          relu_activator), "conv3_3" },
        { std::make_shared<pl_t>(
          2, // window
          2 // stride
          ), "pool3" },
        /* ############## CONVOLUTION LAYER 4 ################# */
        { std::make_shared<cnn_t>(
          shape3d_t(7, 7, 256), // input
          shape3d_t(3, 3, 256), // filter
          512, // filters count
          1, // stride
          padding_type::same,
          relu_activator), "conv4_1" },
        { std::make_shared<cnn_t>(
          shape3d_t(7, 7, 512), // input
          shape3d_t(3, 3, 512), // filter
          512, // filters count
          1, // stride
          padding_type::same,
          relu_activator), "conv4_2" },
        { std::make_shared<cnn_t>(
          shape3d_t(7, 7, 512), // input
          shape3d_t(3, 3, 512), // filter
          512, // filters count
          1, // stride
          padding_type::same,
          relu_activator), "conv4_3" },
        /* ############## FULLY-CONNECTED LAYER ################# */
        { std::make_shared<fc_t>(
          25088, // layer in
          4096, // layer out
          relu_activator), "fc1" },
        { std::make_shared<fc_t>(
          4096, // layer in
          2048, // layer out
          relu_activator), "fc2" },
        { std::make_shared<fc_t>(
          2048, // layer in
          200, // layer out
          relu_activator), "fc3" }
    };

    QString path = testsDir + "/saved_Layers";
    fs_loader_factory<float> factory(path);
    for (auto &l: layers) {
        auto loader = factory.loader(l.first, l.second);
        loader->load();
        l.first->load(loader->weights(), loader->biases());
    }

    layers_list out;
    for (auto &l: layers) {
        out.emplace_back(l.first);
    }
    return out;
}

yannpp::array3d_t<float> read_image(const QString &path) {
    QImage image;
    image.load(path);

    using namespace yannpp;
    array3d_t<float> input(shape3d_t(56, 56, 3), 0.0f);

    for (auto i = 0; i < 56; i++) {
        for (auto j = 0; j < 56; j++) {
            int x = j, y = i;

            auto pixel = image.pixel(y+4, x+4);

            input(x, y, 0) = (qRed(pixel) - 128.f) / 128.f;
            input(x, y, 1) = (qGreen(pixel) - 128.f) / 128.f;
            input(x, y, 2) = (qBlue(pixel) - 128.f)/ 128.f;
        }
    }

    return input;
}

std::vector<std::pair<int, float>> find_top_n_indices(yannpp::array3d_t<float> const &data, int n) {
    using p_t = std::pair<int, float>;
    std::vector<p_t> items;
    const auto &shape = data.shape();
    assert(shape.x() == shape.capacity());
    for (int i = 0; i < data.size(); i++) { items.emplace_back(i, data(i)); }
    std::sort(items.begin(), items.end(), [](p_t const &a, p_t const &b) {
        // reverse sort
        return a.second > b.second;
    });
    std::vector<p_t> result;
    int i = 0;
    for (auto &p: items) { result.push_back(p); if (i++ >= n) { break; } }
    return result;
}

int main(int, char *[])
{
    auto layers = create_layers();
    yannpp::network2_t<float> network(std::move(layers));
    auto input = read_image(testsDir + "/test_4.JPEG");
    //yannpp::log(input);
    auto output = network.feedforward(input);

    parsed_labels_t parsed_labels(
                testsDir + "/wnids.txt",
                testsDir + "/words.txt");
    parsed_labels.read();
    auto top_5 = find_top_n_indices(output, 5);

    std::cout << std::endl;
    for (auto &s: parsed_labels.describe(top_5)) {
        std::cout << s.first << " - " << s.second << std::endl;
    }

    //model->init();
    /*model->load_numbers_from_file();
    model->load_labels();

    model->forward();
    model->top_n(5);

    */
/*
    auto array = cnpy::npz_load("/home/lyubomyr/Projects/tiny_imagenet/multiple_nn/src/test_array.npz", "arr_0");
    auto shape = array.shape;
    auto array_numbers = array.data<double>();

    auto d1 = shape[0];
    auto d2 = shape[1];
    auto d3 = shape[2];
    auto d4 = shape[3];

    QVector<arma::cube> cube_array;
    cube_array.reserve(d4);

    for (auto i = 0; i < d4; i++)
    {
        cube_array.push_back(arma::cube(d1, d2, d3));
    }

    auto index = 0;
    for (auto i = 0; i < d1; i++)
    {
        for (auto j = 0; j < d2; j++)
        {
            for (auto z = 0; z < d3; z++)
            {
                for (auto q = 0; q < d4; q++)
                {
                    cube_array[q](i, j, z) = array_numbers[index];

                    index++;
                }
            }
        }
    }

    for (auto i = 0; i < d3; i++)
    {
        for (auto j = 0; j < d4; j++)
        {
            std::cout << std::fixed << std::setw(8) << std::setfill('0') << std::setprecision(6) << cube_array[j](1, 2, i) << '\t';
        }
        std::cout << std::endl;
    }
*/
}
