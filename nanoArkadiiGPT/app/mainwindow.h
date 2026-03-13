#pragma once

#include <QMainWindow>
#include <QListWidget>
#include <QSpinBox>
#include <QPushButton>
#include <QPlainTextEdit>
#include <QString>
#include <QStringList>

#ifdef slots
#undef slots
#endif

#include <torch/torch.h>
#include <fstream>
#include "../ArkadiiGPT/nanoArkadiiGPT.h"
#include "ui_mainwindow.h"

namespace F = torch::nn::functional;


class MyWindow : public QMainWindow
{
public:
    explicit MyWindow(QWidget* parent = nullptr);
    ~MyWindow();

private:
    Ui::MainWindow ui;
    //Bigram model0;
    ArkadiiGPT model1 = nullptr;
    bool modelLoaded = false;
    void setupModels();
    void setupSpinBox();
    void setupTextEdit();
    QStringList modelPathCandidates() const;
    bool tryLoadModel();
    void onGenerateClicked();
    QString getLastTokens(const QString& text, int maxTokens) const;
    QString runBigramGeneration(const QString& context, int wordsToGenerate);
    QString runArkadiiGeneration(const QString& context, int wordsToGenerate);
};
