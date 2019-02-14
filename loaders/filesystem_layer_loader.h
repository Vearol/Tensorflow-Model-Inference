#ifndef FILESYSTEM_LAYER_LOADER_H
#define FILESYSTEM_LAYER_LOADER_H

#include <vector>

#include <QDir>
#include <QFileInfo>
#include <QString>
#include <QStringList>
#include <QDebug>

#include <cnpy/cnpy.h>

#include <common/array3d.h>
#include <common/shape.h>
#include <common/log.h>

#include "loaders/layer_loader.h"

template<typename T>
class filesystem_layer_loader_t: public layer_loader_t<T> {
public:
    filesystem_layer_loader_t(QString const &layer_directory_path):
        layer_directory_path_(layer_directory_path)
    {}

    // layer_loader_t interface
public:
    virtual std::vector<yannpp::array3d_t<T>> &weights() override { return weights_; }
    virtual std::vector<yannpp::array3d_t<T>> &biases() override { return biases_; }

    virtual void load() override {
        QDir directory(layer_directory_path_);
        auto npz_arrays = directory.entryInfoList(QStringList() << "*.npz");

        for (auto &array_path: npz_arrays) {
            auto full_path = array_path.absoluteFilePath();
            qInfo() << "Processing " << full_path;

            if (full_path.contains("kernel")) {
                initialize_weights(full_path.toStdString());
                continue;
            }

            if (full_path.contains("bias")) {
                initialize_biases(full_path.toStdString());
            }
        }
    }

protected:
    virtual void initialize_biases(const std::string &array_path) {
        auto array = cnpy::npz_load(array_path, "arr_0");
        const size_t shape = array.shape[0];
        auto array_numbers = array.data<T>();
        biases_.emplace_back(yannpp::shape_row(shape), 0);

        for (auto i = 0; i < shape; i++) {
            biases_[0](i) = array_numbers[i];
        }
    }

    virtual void initialize_weights(const std::string &array_path) = 0;

protected:
    QString layer_directory_path_;
    std::vector<yannpp::array3d_t<T>> weights_;
    std::vector<yannpp::array3d_t<T>> biases_;
};

#endif // FILESYSTEM_LAYER_LOADER_H
