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
    layers/functional_layer.cpp

HEADERS += \
    layers/cnn_model.h \
    layers/convolution_layer.h \
    layers/dense_layer.h \
    layers/max_pooling_layer.h \
    layers/relu_layer.h \
    layers/softmax_layer.h \
    layers/layer.h \
    layers/physical_layer.h \
    layers/functional_layer.h

unix:!macx: LIBS += -L$$PWD/../../../../../../../../usr/local/lib/ -lcnpy

INCLUDEPATH += $$PWD/../../../../../../../../usr/local/include
DEPENDPATH += $$PWD/../../../../../../../../usr/local/include
