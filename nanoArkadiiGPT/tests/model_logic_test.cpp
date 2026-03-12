#include <gtest/gtest.h>
#include <torch/torch.h>

#include "config.h"
#include "head/head.h"
#include "multihead_attention/multihead_attention.h"
#include "transformerblock/transformerblock.h"
#include "ArkadiiGPT/nanoArkadiiGPT.h"

namespace {
torch::Tensor make_input(int batch = 2, int time = 4) {
    return torch::randn({batch, time, config.embed_dim_num}, torch::TensorOptions().dtype(torch::kFloat32));
}
}

TEST(ConfigTest, VocabularyAndEncoderAreConsistent) {
    ASSERT_EQ(static_cast<int>(config.vocab.size()), config.vocab_size);
    for (int i = 0; i < static_cast<int>(config.vocab.size()); ++i) {
        ASSERT_TRUE(config.stoi.contains(config.vocab[i]));
        EXPECT_EQ(config.stoi[config.vocab[i]], i);
    }
}

TEST(HeadTest, ForwardReturnsExpectedShape) {
    Head head(config.embed_dim_num / config.head_number);
    head->eval();

    auto x = make_input(2, 5);
    auto out = head->forward(x);

    ASSERT_EQ(out.dim(), 3);
    EXPECT_EQ(out.size(0), 2);
    EXPECT_EQ(out.size(1), 5);
    EXPECT_EQ(out.size(2), config.embed_dim_num / config.head_number);
    EXPECT_TRUE(torch::isfinite(out).all().item<bool>());
}

TEST(HeadTest, CausalMaskPreventsFutureTokenInfluence) {
    Head head(config.embed_dim_num / config.head_number);
    head->eval();

    auto x1 = make_input(1, 4);
    auto x2 = x1.clone();

    x2.index_put_({0, 3}, torch::randn({config.embed_dim_num}));

    auto y1 = head->forward(x1);
    auto y2 = head->forward(x2);

    auto first_three_y1 = y1.index({0, torch::indexing::Slice(0, 3)});
    auto first_three_y2 = y2.index({0, torch::indexing::Slice(0, 3)});

    EXPECT_TRUE(torch::allclose(first_three_y1, first_three_y2, 1e-5, 1e-6));
}

TEST(MultiHeadAttentionTest, ForwardKeepsEmbeddingDimension) {
    MultiHeadAttention mha(config.head_number, config.embed_dim_num / config.head_number);
    mha->eval();

    auto x = make_input(3, 6);
    auto out = mha->forward(x);

    ASSERT_EQ(out.dim(), 3);
    EXPECT_EQ(out.size(0), 3);
    EXPECT_EQ(out.size(1), 6);
    EXPECT_EQ(out.size(2), config.embed_dim_num);
    EXPECT_TRUE(torch::isfinite(out).all().item<bool>());
}

TEST(FeedForwardTest, ForwardKeepsShape) {
    FeedForward ff(config.embed_dim_num);
    ff->eval();

    auto x = make_input(2, 4);
    auto out = ff->forward(x);

    ASSERT_EQ(out.sizes(), x.sizes());
    EXPECT_TRUE(torch::isfinite(out).all().item<bool>());
}

TEST(TransformerBlockTest, ForwardKeepsShape) {
    TransformerBlock block(config.embed_dim_num, config.head_number);
    block->eval();

    auto x = make_input(2, 4);
    auto out = block->forward(x);

    ASSERT_EQ(out.sizes(), x.sizes());
    EXPECT_TRUE(torch::isfinite(out).all().item<bool>());
}

TEST(ArkadiiGPTTest, ForwardReturnsLogitsOfExpectedShape) {
    ArkadiiGPT model;
    model->eval();

    auto idx = torch::randint(
        0,
        config.vocab_size,
        {2, 7},
        torch::TensorOptions().dtype(torch::kLong)
    );

    auto logits = model->forward(idx);

    ASSERT_EQ(logits.dim(), 3);
    EXPECT_EQ(logits.size(0), 2);
    EXPECT_EQ(logits.size(1), 7);
    EXPECT_EQ(logits.size(2), config.vocab_size);
    EXPECT_TRUE(torch::isfinite(logits).all().item<bool>());
}

TEST(ArkadiiGPTTest, ForwardWithTargetsReturnsFlatLogitsAndScalarLoss) {
    ArkadiiGPT model;
    model->eval();

    auto idx = torch::randint(
        0,
        config.vocab_size,
        {2, 5},
        torch::TensorOptions().dtype(torch::kLong)
    );
    auto targets = torch::randint(
        0,
        config.vocab_size,
        {2, 5},
        torch::TensorOptions().dtype(torch::kLong)
    );

    auto result = model->forward(idx, targets);
    auto logits = result.first;
    auto loss = result.second;

    EXPECT_EQ(logits.dim(), 2);
    EXPECT_EQ(logits.size(0), 2 * 5);
    EXPECT_EQ(logits.size(1), config.vocab_size);
    EXPECT_EQ(loss.dim(), 0);
    EXPECT_TRUE(torch::isfinite(loss).item<bool>());
}

TEST(ArkadiiGPTTest, GenerateExtendsSequenceLength) {
    ArkadiiGPT model;
    model->eval();

    auto idx = torch::randint(
        0,
        config.vocab_size,
        {1, 3},
        torch::TensorOptions().dtype(torch::kLong)
    );

    auto out = model->generate(idx, 4);

    EXPECT_EQ(out.size(0), 1);
    EXPECT_EQ(out.size(1), 7);
}

TEST(BatchTest, GetBatchReturnsCorrectShapesAndShiftedTargets) {
    auto data = torch::arange(
        0,
        config.block_size + 100,
        torch::TensorOptions().dtype(torch::kInt64)
    );

    auto batch = get_batch(data);
    auto x = batch.first;
    auto y = batch.second;

    EXPECT_EQ(x.size(0), config.batch_size);
    EXPECT_EQ(x.size(1), config.block_size);
    EXPECT_EQ(y.size(0), config.batch_size);
    EXPECT_EQ(y.size(1), config.block_size);

    auto x_shifted = x.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)});
    auto y_without_last = y.index({torch::indexing::Slice(), torch::indexing::Slice(0, config.block_size - 1)});

    EXPECT_TRUE(torch::equal(x_shifted, y_without_last));
}
