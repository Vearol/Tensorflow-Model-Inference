#include <QCoreApplication>
#include <QDebug>

#include <armadillo>

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    arma::cube B(3, 3, 3, arma::fill::zeros);

    B(0,0,0) = 5;

    qInfo() << B(0,0,0);

    return a.exec();
}
