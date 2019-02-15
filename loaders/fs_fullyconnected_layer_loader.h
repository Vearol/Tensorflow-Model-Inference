#ifndef FS_FULLYCONNECTED_LAYER_LOADER_H
#define FS_FULLYCONNECTED_LAYER_LOADER_H

#include <string>

#include <QString>

#include <yannpp/common/array3d.h>
#include <yannpp/common/shape.h>

#include "filesystem_layer_loader.h"

template<typename T>
class fs_fullyconnected_layer_loader_t: public filesystem_layer_loader_t<T> {
public:
    fs_fullyconnected_layer_loader_t(QString const &layer_directory_path):
        filesystem_layer_loader_t<T>(layer_directory_path)
    {}

protected:
    virtual void initialize_weights(const std::string &array_path) override {
        auto array = cnpy::npz_load(array_path, "arr_0");
        auto shape = array.shape;
        auto array_numbers = array.data<T>();

        auto input_flatten_size = shape[0];
        auto output_size = shape[1];

        // `tf.dense` behaves by contracting last index of input tensor with first index of weights tensor
        this->weights_.emplace_back(yannpp::shape3d_t(output_size, input_flatten_size, 1), 0);

        auto index = 0;
        for (auto height = 0; height < input_flatten_size; height++) {
            for (auto width = 0; width < output_size; width++) {
                this->weights_[0](width, height) = array_numbers[index++];
            }
        }
    }
};

#endif // FS_FULLYCONNECTED_LAYER_LOADER_H
