QT += gui

CONFIG += console
CONFIG -= app_bundle
CONFIG += c++11

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS
DEFINES += QT_MESSAGELOGCONTEXT

INCLUDEPATH += $$(PWD)
INCLUDEPATH += yannpp/src

# zlib for cnpy
LIBS += -lz

win32 {
    DEFINES += NOMINMAX
    DEFINES += ZLIB_WINAPI
    INCLUDEPATH += vendors/zlib
    LIBS += -L"$$PWD/vendors/zlib"
    LIBS += -lz
}

TESTS_DIR = $$PWD/test
DEFINES += TESTS_DIR=$${TESTS_DIR}

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += main.cpp \
    yannpp/src/yannpp/common/utils.cpp \
    yannpp/src/yannpp/common/log.cpp \
    yannpp/src/yannpp/common/cpphelpers.cpp \
    parsing/parsed_labels.cpp \
    cnpy/cnpy.cpp

HEADERS += \
    yannpp/src/yannpp/common/utils.h \
    yannpp/src/yannpp/common/shape.h \
    yannpp/src/yannpp/common/log.h \
    yannpp/src/yannpp/common/cpphelpers.h \
    yannpp/src/yannpp/common/array3d_math.h \
    yannpp/src/yannpp/common/array3d.h \
    yannpp/src/yannpp/layers/poolinglayer.h \
    yannpp/src/yannpp/layers/layer_base.h \
    yannpp/src/yannpp/layers/layer_metadata.h \
    yannpp/src/yannpp/layers/fullyconnectedlayer.h \
    yannpp/src/yannpp/layers/crossentropyoutputlayer.h \
    yannpp/src/yannpp/layers/convolutionlayer.h \
    yannpp/src/yannpp/network/network2.h \
    yannpp/src/yannpp/network/activator.h \
    loaders/layer_loader.h \
    loaders/filesystem_layer_loader.h \
    loaders/fs_convolution_layer_loader.h \
    loaders/fs_fullyconnected_layer_loader.h \
    loaders/fs_loader_factory.h \
    parsing/parsed_labels.h \
    cnpy/cnpy.h

