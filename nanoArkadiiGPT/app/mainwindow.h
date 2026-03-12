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

struct HeadImpl : torch::nn::Module {
    torch::nn::Linear query = nullptr;
    torch::nn::Linear key = nullptr;
    torch::nn::Linear value = nullptr;
    torch::Tensor mask = torch::Tensor(nullptr);
    torch::nn::Dropout drop = nullptr;

    HeadImpl() = default;
    HeadImpl(int head_size) {
        mask = register_buffer("mask", torch::tril(torch::ones({block_size, block_size})));
        query = register_module(
            "query",
            torch::nn::Linear(torch::nn::LinearOptions(embed_dim_num, head_size).bias(false))
            );
        key = register_module(
            "key",
            torch::nn::Linear(torch::nn::LinearOptions(embed_dim_num, head_size).bias(false))
            );
        value = register_module(
    "value",
    torch::nn::Linear(torch::nn::LinearOptions(embed_dim_num, head_size).bias(false))
    );
        drop = register_module(
            "drop",
            torch::nn::Dropout(dropout)
            );
    }

    torch::Tensor forward(torch::Tensor& x) {
        auto T = x.size(1);
        auto K = key(x);
        auto Q = query(x);
        auto V = value(x);
        torch::Tensor w = Q.matmul(K.transpose(-2, -1));
        w = w / sqrt(K.size(-1));
        w = w.masked_fill(
            mask.index({torch::indexing::Slice(0, T), torch::indexing::Slice(0, T)}) == 0,
            float(-INFINITY)
            );
        w = F::softmax(w, F::SoftmaxFuncOptions(-1));
        w = drop(w);
        auto res = w.matmul(V);
        return res;

    }
};
TORCH_MODULE(Head);

struct MultiHeadAttentionImpl : torch::nn::Module {
    torch::nn::ModuleList heads = nullptr;
    torch::nn::Linear projection = nullptr;
    torch::nn::Dropout drop = nullptr;
    MultiHeadAttentionImpl(int heads_amount, int head_size) {
        heads = register_module(
            "heads",
            torch::nn::ModuleList()
            );
        for (int i = 0; i < heads_amount; ++i) {
            heads->push_back(Head(head_size));
        }
        projection = register_module(
            "projection",
            torch::nn::Linear(embed_dim_num, embed_dim_num)
            );
        drop = register_module(
            "drop",
            torch::nn::Dropout(dropout)
            );
    }
    MultiHeadAttentionImpl() = default;
    torch::Tensor forward(torch::Tensor x) {
        std::vector<torch::Tensor> rawres;
        for (auto& raw : *heads) {
            auto h = raw->as<Head>();
            rawres.push_back(h->forward(x));
        }
        return drop(projection(torch::cat(rawres, -1)));
    }
};
TORCH_MODULE(MultiHeadAttention);

struct FeedForwardImpl : torch::nn::Module {
    torch::nn::Sequential mod = nullptr;
    FeedForwardImpl() = default;
    FeedForwardImpl(int embed_dim_num) {
        mod = register_module(
            "mod",
            torch::nn::Sequential(
                torch::nn::Linear(embed_dim_num, embed_dim_num * 4),
                torch::nn::ReLU(),
                torch::nn::Linear(embed_dim_num * 4, embed_dim_num),
                torch::nn::Dropout(dropout)
                )
            );
    }
    torch::Tensor forward(torch::Tensor x) {
        return mod->forward(x);
    }
};
TORCH_MODULE(FeedForward);

struct TransformerBlockImpl : torch::nn::Module {
    MultiHeadAttention att = nullptr;
    FeedForward feed = nullptr;
    torch::nn::LayerNorm ln1 = nullptr;
    torch::nn::LayerNorm ln2 = nullptr;

    TransformerBlockImpl() = default;

    TransformerBlockImpl(int embed_dim_num, int head_amount) {
        int head_size = embed_dim_num / head_amount;
        att = register_module(
            "att",
            MultiHeadAttention(head_amount, head_size)
            );
        feed = register_module(
            "feed",
            FeedForward(embed_dim_num)
            );
        ln1 = register_module(
            "ln1",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim_num}))
            );
        ln2 = register_module(
        "ln2",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim_num}))
            );
    }
    torch::Tensor forward(torch::Tensor x) {
        x = x + att(ln1(x));
        x = x + feed(ln2(x));
        return x;
    }
};
TORCH_MODULE(TransformerBlock);


struct ArkadiiGPTImpl : torch::nn::Module {

    torch::nn::Embedding embedding_table = nullptr;
    torch::nn::Linear main_head = nullptr;
    torch::nn::Embedding position_embedding = nullptr;
    torch::nn::Sequential transformers = nullptr;

    /// @param vocab_size -
    ArkadiiGPTImpl() {
        embedding_table = register_module(
            "embedding_table",
            torch::nn::Embedding(vocab_size, embed_dim_num)
            );
        main_head = register_module(
            "main_head",
            torch::nn::Linear(embed_dim_num, vocab_size)
            );
        position_embedding = register_module(
            "position_embedding",
            torch::nn::Embedding(block_size, embed_dim_num)
            );
        transformers = register_module(
            "transformers",
            torch::nn::Sequential(
                TransformerBlock(embed_dim_num, head_number),
                TransformerBlock(embed_dim_num, head_number),
                TransformerBlock(embed_dim_num, head_number),
                TransformerBlock(embed_dim_num, head_number),
                TransformerBlock(embed_dim_num, head_number),
                TransformerBlock(embed_dim_num, head_number),
                torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim_num}))
            )
            );
    }
    torch::Tensor forward(torch::Tensor& idx) {
        auto idxTime = idx.size(1);
        auto tokens_embed = embedding_table(idx);
        auto position_embed = position_embedding(
            torch::arange(
                idxTime,
                torch::TensorOptions().dtype(torch::kLong).device(idx.device())
                )
                );
        auto X = tokens_embed + position_embed;
        X = transformers->forward(X);
        auto logits = main_head(X);
        return logits;
    }

    ///
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& idx, torch::Tensor& targets) {
        auto idxTime = idx.size(1);
        auto tokens_embed = embedding_table(idx);
        auto position_embed = position_embedding(
            torch::arange(
                idxTime,
                torch::TensorOptions().dtype(torch::kLong).device(idx.device())
                )
                );
        auto X = tokens_embed + position_embed;
        X = transformers->forward(X);
        auto logits = main_head(X);
        auto B = logits.size(0);
        auto T = logits.size(1);
        auto C = logits.size(2);
        logits = logits.view({B*T, C});
        targets = targets.view({B*T});
        auto loss = F::cross_entropy(logits, targets);
        return {logits, loss};
    }

    torch::Tensor generate(torch::Tensor& idx, int max_new_tokens) {
        for (int i = 0; i < max_new_tokens; ++i) {
            auto idx_sliced = idx.index({
                torch::indexing::Slice(),
                torch::indexing::Slice(-block_size, torch::indexing::None)
            });
            auto logits = forward(idx_sliced);
            logits = logits.index({torch::indexing::Slice(), -1, torch::indexing::Slice()});
            auto probs = F::softmax(logits, F::SoftmaxFuncOptions(-1));
            auto pred = torch::multinomial(probs, 1);
            //auto pred = torch::argmax(probs, 1).unsqueeze(1);
            idx = torch::cat({idx, pred}, 1);
        }
        return idx;
    }
};

TORCH_MODULE(ArkadiiGPT);

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
