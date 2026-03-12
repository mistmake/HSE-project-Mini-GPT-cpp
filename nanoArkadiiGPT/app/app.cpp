#include <QApplication>
#include "mainwindow.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    for (int i = 0; i < vocab.size(); ++i) {
        stoi[vocab[i]] = i;
    }

    MyWindow window;
    window.show();

    return app.exec();
}