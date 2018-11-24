#ifndef FS_CONVOLUTION_LAYER_LOADER_H
#define FS_CONVOLUTION_LAYER_LOADER_H

#include <string>

#include <QString>

#include <common/array3d.h>
#include <common/shape.h>

#include "filesystem_layer_loader.h"

template<typename T>
class fs_convolution_layer_loader_t: public filesystem_layer_loader_t<T> {
public:
    fs_convolution_layer_loader_t(QString const &layer_directory_path):
        filesystem_layer_loader_t<T>(layer_directory_path)
    {}

protected:
    virtual void initialize_weights(const std::string &array_path) override {
        auto array = cnpy::npz_load(array_path, "arr_0");
        auto shape = array.shape;
        auto array_numbers = array.data<T>();

        auto filter_width = shape[0];
        auto filter_height = shape[1];
        auto input_depth = shape[2];
        auto filters_number = shape[3];

        this->weights_.reserve(filters_number);

        for (auto filter = 0; filter < filters_number; filter++) {
            this->weights_.emplace_back(yannpp::shape3d_t(filter_height, filter_width, input_depth), 0);
        }

        auto index = 0;
        for (auto height = 0; height < filter_height; height++) {
            for (auto width = 0; width < filter_width; width++) {
                for (auto depth = 0; depth < input_depth; depth++) {
                    for (auto filter = 0; filter < filters_number; filter++, index++) {
                        this->weights_[filter](height, width, depth) = array_numbers[index];
                    }
                }
            }
        }
    }

    virtual void initialize_biases(const std::string &array_path) override {
        auto array = cnpy::npz_load(array_path, "arr_0");
        size_t shape = array.shape[0];
        auto array_numbers = array.data<T>();
        this->biases_.reserve(shape);

        for (auto i = 0; i < shape; i++) {
            this->biases_.emplace_back(yannpp::shape_row(1), array_numbers[i]);
        }
    }
};

#endif // FS_CONVOLUTION_LAYER_LOADER_H
