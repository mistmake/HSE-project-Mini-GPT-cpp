#pragma once

#include <QMainWindow>
#include <QListWidget>
#include <QSpinBox>
#include <QPushButton>
#include <QPlainTextEdit>
#include <QString>

#ifdef slots
#undef slots
#endif

#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <map>
#include <cmath>
#include "ui_mainwindow.h"

namespace F = torch::nn::functional;

constexpr int
batch_size = 64,
block_size = 256,
vocab_size = 65,
learning_amount = 50000,
embed_dim_num = 384, // n_embd
head_number = 6;
constexpr float dropout = 0.2;
torch::Device device(torch::kCPU);

int version = 0;

std::string vocab = " !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

std::map<char, int> stoi;


class MyWindow : public QMainWindow
{
public:
    explicit MyWindow(QWidget* parent = nullptr)
        : QMainWindow(parent)
    {
        std::cout << "constructing";
        ui.setupUi(this);

        setupModels();
        setupSpinBox();
        setupTextEdit();
        model1 = ArkadiiGPT();
        connect(ui.pushButton, &QPushButton::clicked, this, [this]()
        {
            onGenerateClicked();
        });
        std::cout << "loading";
        torch::load(model1, "../nanoArkadiiGPT/model.pt", device);
        model1->eval();
        std::cout << "loaded";
    }

    ~MyWindow()
    {
    }

private:
    Ui::MainWindow ui;
    //Bigram model0;
    ArkadiiGPT model1 = nullptr;

private:
    void setupModels()
    {

        ui.listWidget->setCurrentRow(1);
    }

    void setupSpinBox()
    {
        ui.spinBox->setMinimum(1);
        ui.spinBox->setMaximum(1000);
        ui.spinBox->setValue(50);
    }

    void setupTextEdit()
    {
    }

    void onGenerateClicked()
    {
        QListWidgetItem* currentItem = ui.listWidget->currentItem();
        if (!currentItem)
        {
            return;
        }

        const QString fullText = ui.plainTextEdit->toPlainText();
        const QString context = getLastTokens(fullText, 256);
        const int wordsToGenerate = ui.spinBox->value();
        const int selectedRow = ui.listWidget->currentRow();


        QString generatedText;

        if (selectedRow == 0)
        {

            //generatedText = runBigramGeneration(context, wordsToGenerate);
        }
        else if (selectedRow == 1)
        {
            generatedText = runArkadiiGeneration(context, wordsToGenerate);
        }

        QString finalText = fullText;
        finalText += generatedText;
        ui.plainTextEdit->setPlainText(finalText);
    }


    QString getLastTokens(const QString& text, int maxTokens) const
    {
        if (text.size() <= maxTokens)
            return text;

        return text.right(maxTokens);
    }

    QString runBigramGeneration(const QString& context, int wordsToGenerate)
    {
        std::string input = context.toStdString();

        return QString();
    }

    QString runArkadiiGeneration(const QString& context, int wordsToGenerate)
    {
        std::string input = context.toStdString();
        std::vector<int> coded;

        for (char c : input) {
            coded.push_back(stoi[c]);
        }
        if (coded.empty()) {
            coded.push_back(0);
        }
        torch::Tensor batch = torch::tensor(coded).view({1, -1});
        std::string res;
        for (int i = 0; i < wordsToGenerate; ++i) {
            while (true) {
                model1->generate(batch, 1);
                res.push_back(vocab[batch[0][-1].item<int>()]);
                if (batch[0][-1].item<int64_t>() == 0) {
                    break;
                }
            }
        }

        return QString::fromStdString(res);
    }
};
