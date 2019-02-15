#include <algorithm>
#include <iostream>
#include <iomanip>
#include <future>
#include <vector>
#include <memory>
#include <utility>

#include <QCoreApplication>
#include <QDebug>
#include <QImage>
#include <QtGlobal>

#include <cnpy/cnpy.h>

#include <yannpp/common/array3d.h>
#include <yannpp/common/array3d_math.h>
#include <yannpp/common/log.h>
#include <yannpp/layers/layer_base.h>
#include <yannpp/layers/layer_metadata.h>
#include <yannpp/layers/convolutionlayer.h>
#include <yannpp/layers/fullyconnectedlayer.h>
#include <yannpp/layers/poolinglayer.h>
#include <yannpp/network/network2.h>
#include <yannpp/network/activator.h>

#include "loaders/fs_loader_factory.h"
#include "parsing/parsed_labels.h"

#define STRINGIZE_(x) #x
#define STRINGIZE(x) STRINGIZE_(x)

const QString testsDir = STRINGIZE(TESTS_DIR);

using layers_list = std::vector<std::shared_ptr<yannpp::layer_base_t<float>>>;

static yannpp::activator_t<float> relu_activator(yannpp::relu_v<float>, yannpp::relu_v<float>);

layers_list create_layers() {
    using namespace yannpp;

    using cnn_t = convolution_layer_t<float>;
    using fc_t = fully_connected_layer_t<float>;
    using pl_t = pooling_layer_t<float>;
    using m_t = layer_metadata_t;

    qInfo() << "Creating layers...";

    layers_list layers = {
        /* ############## CONVOLUTION LAYER 1 ################# */
        std::make_shared<cnn_t>(
        shape3d_t(56, 56, 3), // input
        shape3d_t(3, 3, 3), // filter
        64, // filters count
        1, // stride
        padding_type::same,
        relu_activator,
        m_t{ "conv1_1" }),
        std::make_shared<cnn_t>(
        shape3d_t(56, 56, 64), // input
        shape3d_t(3, 3, 64), // filter
        64, // filters count
        1, // stride
        padding_type::same,
        relu_activator,
        m_t{ "conv1_2" }),
        std::make_shared<pl_t>(
        2, // window
        2, // stride
        m_t{ "pool1" }),
        /* ############## CONVOLUTION LAYER 2 ################# */
        std::make_shared<cnn_t>(
        shape3d_t(28, 28, 64), // input
        shape3d_t(3, 3, 64), // filter
        128, // filters count
        1, // stride
        padding_type::same,
        relu_activator,
        m_t{ "conv2_1" }),
        std::make_shared<cnn_t>(
        shape3d_t(28, 28, 128), // input
        shape3d_t(3, 3, 128), // filter
        128, // filters count
        1, // stride
        padding_type::same,
        relu_activator,
        m_t{ "conv2_2" }),
        std::make_shared<pl_t>(
        2, // window
        2, // stride
        m_t{ "pool2" }),
        /* ############## CONVOLUTION LAYER 3 ################# */
        std::make_shared<cnn_t>(
        shape3d_t(14, 14, 128), // input
        shape3d_t(3, 3, 128), // filter
        256, // filters count
        1, // stride
        padding_type::same,
        relu_activator,
        m_t{ "conv3_1" }),
        std::make_shared<cnn_t>(
        shape3d_t(14, 14, 256), // input
        shape3d_t(3, 3, 256), // filter
        256, // filters count
        1, // stride
        padding_type::same,
        relu_activator,
        m_t{ "conv3_2" }),
        std::make_shared<cnn_t>(
        shape3d_t(14, 14, 256), // input
        shape3d_t(3, 3, 256), // filter
        256, // filters count
        1, // stride
        padding_type::same,
        relu_activator,
        m_t{ "conv3_3" }),
        std::make_shared<pl_t>(
        2, // window
        2, // stride
        m_t{ "pool3" }),
        /* ############## CONVOLUTION LAYER 4 ################# */
        std::make_shared<cnn_t>(
        shape3d_t(7, 7, 256), // input
        shape3d_t(3, 3, 256), // filter
        512, // filters count
        1, // stride
        padding_type::same,
        relu_activator,
        m_t{ "conv4_1" }),
        std::make_shared<cnn_t>(
        shape3d_t(7, 7, 512), // input
        shape3d_t(3, 3, 512), // filter
        512, // filters count
        1, // stride
        padding_type::same,
        relu_activator,
        m_t{ "conv4_2" }),
        std::make_shared<cnn_t>(
        shape3d_t(7, 7, 512), // input
        shape3d_t(3, 3, 512), // filter
        512, // filters count
        1, // stride
        padding_type::same,
        relu_activator,
        m_t{ "conv4_3" }),
        /* ############## FULLY-CONNECTED LAYER ################# */
        std::make_shared<fc_t>(
        25088, // layer in
        4096, // layer out
        relu_activator,
        m_t{ "fc1" }),
        std::make_shared<fc_t>(
        4096, // layer in
        2048, // layer out
        relu_activator,
        m_t{ "fc2" }),
        std::make_shared<fc_t>(
        2048, // layer in
        200, // layer out
        relu_activator,
        m_t{ "fc3" })
    };

    qInfo() << "Loading layers...";

    std::vector<std::future<void>> futures;

    fs_loader_factory<float> factory(testsDir + "/saved_Layers");
    const size_t size = layers.size();
    for (size_t i = 0; i < size; i++) {
        auto layer = layers[i];
        auto loader = factory.loader(layer);
        futures.emplace_back(std::async(std::launch::async, [loader, layer](){
            loader->load();
            layer->load(std::move(loader->weights()), std::move(loader->biases()));
        }));
    }

    for (auto &f: futures) {
        f.get();
    }

    return layers;
}

yannpp::array3d_t<float> read_image(const QString &path) {
    qInfo() << "Reading image" << path;

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

int main(int, char *[]) {
    qSetMessagePattern("%{time hh:mm:ss.zzz} %{type} T#%{threadid} %{function} - %{message}");

    auto layers = create_layers();
    yannpp::network2_t<float> network(std::move(layers));
    auto input = read_image(testsDir + "/test_4.JPEG");

    qInfo() << "Running inference...";
    auto output = network.feedforward(input);

    qInfo() << "Output ready";
    yannpp::log(output);

    parsed_labels_t parsed_labels(
                testsDir + "/wnids.txt",
                testsDir + "/words.txt");
    parsed_labels.read();
    auto top_5 = find_top_n_indices(output, 5);

    std::cout << std::endl;
    for (auto &s: parsed_labels.describe(top_5)) {
        std::cout << s.first << " - " << s.second << std::endl;
    }
}
