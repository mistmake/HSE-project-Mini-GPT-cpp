#include "mainwindow.h"
#include "../config.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QStatusBar>

#include <c10/util/Exception.h>
#include <iostream>
#include <vector>
#include <string>

MyWindow::MyWindow(QWidget* parent)
    : QMainWindow(parent)
{
    std::cout << "constructing" << std::endl;

    ui.setupUi(this);

    setupModels();
    setupSpinBox();
    setupTextEdit();

    model1 = ArkadiiGPT();

    connect(ui.pushButton, &QPushButton::clicked, this, [this]()
    {
        onGenerateClicked();
    });

    modelLoaded = tryLoadModel();
    ui.pushButton->setEnabled(modelLoaded);
}

MyWindow::~MyWindow()
{
}

void MyWindow::setupModels()
{
    ui.listWidget->setCurrentRow(1);
}

void MyWindow::setupSpinBox()
{
    ui.spinBox->setMinimum(1);
    ui.spinBox->setMaximum(1000);
    ui.spinBox->setValue(50);
}

void MyWindow::setupTextEdit()
{
    ui.plainTextEdit->setPlaceholderText("Enter prompt here...");
}

QStringList MyWindow::modelPathCandidates() const
{
    const QString appDir = QCoreApplication::applicationDirPath();
    const QString currentDir = QDir::currentPath();

    return {
        QDir(appDir).filePath("../nanoArkadiiGPT/model.pt"),
        QDir(appDir).filePath("nanoArkadiiGPT/model.pt"),
        QDir(currentDir).filePath("../nanoArkadiiGPT/model.pt"),
        QDir(currentDir).filePath("nanoArkadiiGPT/model.pt")
    };
}

bool MyWindow::tryLoadModel()
{
    for (const QString& candidate : modelPathCandidates())
    {
        QFileInfo info(candidate);
        if (!info.exists() || !info.isFile())
        {
            continue;
        }

        try
        {
            std::cout << "loading model from " << info.absoluteFilePath().toStdString() << std::endl;
            torch::load(model1, info.absoluteFilePath().toStdString(), config.device);
            model1->eval();
            statusBar()->showMessage(QString("Model loaded: %1").arg(info.absoluteFilePath()));
            std::cout << "loaded" << std::endl;
            return true;
        }
        catch (const c10::Error& error)
        {
            statusBar()->showMessage(QString("Failed to load model: %1").arg(error.what()));
            return false;
        }
    }

    statusBar()->showMessage("model.pt not found. App started without generation model.");
    return false;
}

void MyWindow::onGenerateClicked()
{
    if (!modelLoaded)
    {
        statusBar()->showMessage("Generation is unavailable until model.pt is added.");
        return;
    }

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
        // generatedText = runBigramGeneration(context, wordsToGenerate);
    }
    else if (selectedRow == 1)
    {
        generatedText = runArkadiiGeneration(context, wordsToGenerate);
    }

    QString finalText = fullText;
    finalText += generatedText;
    ui.plainTextEdit->setPlainText(finalText);
}

QString MyWindow::getLastTokens(const QString& text, int maxTokens) const
{
    if (text.size() <= maxTokens)
        return text;

    return text.right(maxTokens);
}

QString MyWindow::runBigramGeneration(const QString& context, int wordsToGenerate)
{
    std::string input = context.toStdString();
    return QString();
}

QString MyWindow::runArkadiiGeneration(const QString& context, int wordsToGenerate)
{
    std::string input = context.toStdString();
    std::vector<int> coded;

    for (char c : input) {
        coded.push_back(config.stoi[c]);
    }

    if (coded.empty()) {
        coded.push_back(0);
    }

    torch::Tensor batch = torch::tensor(coded).view({1, -1});
    std::string res;

    for (int i = 0; i < wordsToGenerate; ++i) {
        size_t n = 0;
        while (true ) {
            batch = model1->generate(batch, 1);
            res.push_back(config.vocab[batch[0][-1].item<int>()]);
            if (batch[0][-1].item<int64_t>() == 0 || n >= config.max_cpw) {
                break;
            }
            n++;
        }
    }

    return QString::fromStdString(res);
}
