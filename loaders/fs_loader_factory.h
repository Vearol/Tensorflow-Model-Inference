#ifndef FS_LOADER_FACTORY_H
#define FS_LOADER_FACTORY_H

#include <memory>
#include <string>

#include <QString>
#include <QDir>

#include <yannpp/layers/layer_base.h>
#include <yannpp/layers/layer_metadata.h>
#include <yannpp/layers/convolutionlayer.h>
#include <yannpp/layers/fullyconnectedlayer.h>

#include "loaders/layer_loader.h"
#include "loaders/fs_fullyconnected_layer_loader.h"
#include "loaders/fs_convolution_layer_loader.h"

template<typename T>
class fs_loader_factory {
public:
    fs_loader_factory(QString const &root):
        root_dir_(root)
    {}

private:
    class dummy_loader_t: public layer_loader_t<T> {
    public:
        virtual void load() {}
        virtual std::vector<yannpp::array3d_t<T>> &weights() { return empty_; }
        virtual std::vector<yannpp::array3d_t<T>> &biases() { return empty_; }
    private:
        std::vector<yannpp::array3d_t<T>> empty_ = {};
    };

public:
    std::shared_ptr<layer_loader_t<T>> loader(std::shared_ptr<yannpp::layer_base_t<T>> const &layer) {
        auto fc_layer = std::dynamic_pointer_cast<yannpp::fully_connected_layer_t<T>>(layer);
        auto cn_layer = std::dynamic_pointer_cast<yannpp::convolution_layer_base_t<T>>(layer);
        auto &layer_name = layer->get_metadata().name;
        if (fc_layer != nullptr) {
            return fc(layer_name);
        } else if (cn_layer != nullptr) {
            return cn(layer_name);
        } else {
            return std::make_shared<dummy_loader_t>();
        }
    }

private:
    std::shared_ptr<layer_loader_t<T>> fc(std::string const &layer_name) {
        return std::make_shared<fs_fullyconnected_layer_loader_t<T>>(
                    root_dir_.filePath(QString::fromStdString(layer_name)));
    }

    std::shared_ptr<layer_loader_t<T>> cn(std::string const &layer_name) {
        return std::make_shared<fs_convolution_layer_loader_t<T>>(
                    root_dir_.filePath(QString::fromStdString(layer_name)));
    }

private:
    QDir root_dir_;
};

#endif // FS_LOADER_FACTORY_H
