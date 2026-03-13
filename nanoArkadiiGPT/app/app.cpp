#include <QApplication>
#include "mainwindow.h"
#include "../config.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    for (int i = 0; i < config.vocab.size(); ++i) {
        config.stoi[config.vocab[i]] = i;
    }

    MyWindow window;
    window.show();

    return app.exec();
}
