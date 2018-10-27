#include <QCoreApplication>
#include <QDebug>

#include "layers/vgg16.h"

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    QString path = "/home/lyubomyr/Projects/tiny_imagenet/multiple_nn/src/modelExport/saved_Layers";
    CNN_Model* model = new VGG16(path);

    model->init();
    model->load_numbers_from_file();
    model->load_labels();

    model->forward("/home/lyubomyr/Projects/tiny_imagenet/multiple_nn/tiny-imagenet-200/test/images/test_4.JPEG");
    model->top_n(5);

    return a.exec();
}
