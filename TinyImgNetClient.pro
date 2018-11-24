QT += gui

LIBS += \
    -llapack \
    -larmadillo \
    -lblas \
    -L$$PWD/../../../../../../../../usr/local/lib/ -lcnpy

CONFIG += c++11 console
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

INCLUDEPATH += yannpp/src

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += main.cpp \
    layers/cnn_model.cpp \
    layers/convolution_layer.cpp \
    layers/dense_layer.cpp \
    layers/max_pooling_layer.cpp \
    layers/relu_layer.cpp \
    layers/softmax_layer.cpp \
    layers/layer.cpp \
    layers/physical_layer.cpp \
    layers/functional_layer.cpp \
    layers/vgg16.cpp \
    yannpp/src/common/utils.cpp \
    yannpp/src/common/log.cpp \
    yannpp/src/common/cpphelpers.cpp \
    parsing/parsed_labels.cpp

HEADERS += \
    layers/cnn_model.h \
    layers/convolution_layer.h \
    layers/dense_layer.h \
    layers/max_pooling_layer.h \
    layers/relu_layer.h \
    layers/softmax_layer.h \
    layers/layer.h \
    layers/physical_layer.h \
    layers/functional_layer.h \
    layers/vgg16.h \
    yannpp/src/common/utils.h \
    yannpp/src/common/shape.h \
    yannpp/src/common/log.h \
    yannpp/src/common/cpphelpers.h \
    yannpp/src/common/array3d_math.h \
    yannpp/src/common/array3d.h \
    yannpp/src/layers/poolinglayer.h \
    yannpp/src/layers/layer_base.h \
    yannpp/src/layers/fullyconnectedlayer.h \
    yannpp/src/layers/crossentropyoutputlayer.h \
    yannpp/src/layers/convolutionlayer.h \
    yannpp/src/network/network2.h \
    yannpp/src/network/activator.h \
    loaders/layer_loader.h \
    loaders/filesystem_layer_loader.h \
    loaders/fs_convolution_layer_loader.h \
    loaders/fs_fullyconnected_layer_loader.h \
    loaders/fs_loader_factory.h \
    parsing/parsed_labels.h

unix:!macx: LIBS += -L$$PWD/../../../../../../../../usr/local/lib/ -lcnpy

INCLUDEPATH += $$PWD/../../../../../../../../usr/local/include
DEPENDPATH += $$PWD/../../../../../../../../usr/local/include
