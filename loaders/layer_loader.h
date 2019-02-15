#ifndef LAYER_LOADER_H
#define LAYER_LOADER_H

#include <vector>

#include <yannpp/common/array3d.h>

template<typename T>
class layer_loader_t {
public:
    virtual ~layer_loader_t(){}
    virtual void load() = 0;
    virtual std::vector<yannpp::array3d_t<T>> &weights() = 0;
    virtual std::vector<yannpp::array3d_t<T>> &biases() = 0;
};

#endif // LAYER_LOADER_H
