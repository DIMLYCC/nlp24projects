# NLP项目：中文-英文机器翻译
## 项目概述：
项目分别实现了基于LSTM、transformer的seq2seq的翻译模型，并对transformer模型完成了训练。
## 分词

在本项目中，分词是构建机器翻译模型的一个关键步骤。有效的分词方法能够提高词汇表的覆盖率，减少未覆盖词汇的出现，从而优化模型的训练和翻译效果。以下是分词部分尝试的几种方法。

### 1. 基于规则的分词方法

在最初的阶段，尝试了两种基于规则的分词方法：对中文使用**jieba分词**，对英文使用**nltk分词**，或者直接使用字符串的`split()`方法来进行分词。对于中文，jieba是一个广泛使用的中文分词工具，它通过字典和基于词频的词典构建方式，能够识别常见的词汇。然而，这种方法存在以下问题：

- **词汇表过大**：由于jieba是基于词典和统计方法构建的分词模型，词汇表较大，并且容易受限于词典中的词汇。当遇到未登录词或低频词时，jieba的表现可能并不好。
- **未覆盖词汇**：尽管jieba能够处理大部分常见词汇，但对于一些新词或领域特定词汇，模型的分词效果依然有限。

对于英文，使用了nltk的word_tokenize方法。nltk提供的分词工具通常能够准确地处理英文的标点、缩写等细节，但是在处理一些特殊词汇（例如复合词、专有名词等）时，nltk的分词效果也可能不足。

### 2. 逐字符分词方法

在尝试了基于规则的分词后，我们进一步尝试了**逐字符分词**的方法。该方法的核心思想是将每个中文字符都视作一个独立的单位进行处理，数字和外文字符不做分词，直接保留。

逐字符分词能够避免词汇表过大或未覆盖词汇的问题，因为每个字符都是独立的单位。这种方法适用于中文分词时，能够处理未登录词，并且模型能够通过逐字符学习到每个字的特征。但在中小规模数据集上，逐字符分词的效果往往不尽如人意。由于没有明显的语法结构和语义信息，模型难以捕捉到句子中的语义模式。这样会导致模型训练时陷入局部最优解，使得大部分词汇被映射到最常见的字符或词语上。而对于一些不常见的词汇或新词，模型可能无法准确翻译。

### 3. 使用SentencePiece (SPM) 进行分词

为了解决上述方法中的词汇表过大、未覆盖词汇等问题，最终选择了使用**SentencePiece (SPM) 模型**来进行中文和英文的分词。SPM模型是一个无监督的文本预处理工具，通过学习一个共享的子词词汇表来进行分词。SPM并不像传统的分词方法（如基于规则的分词工具）那样依赖于人工设置的词汇表，而是通过统计和学习得到一个最优的子词（subword）词汇表，从而平衡了词汇表大小和未覆盖词汇的出现。

#### SPM分词原理

SentencePiece的核心思想是将整个语料看作一个大字符串，并通过统计方法构建一个字典。这个字典是基于子词单元构建的，而不是基于传统的词汇。SPM将文本切分为若干个子词单元，通常这些子词单元是比字、词级别更细的粒度。SPM的好处在于它能够通过处理所有输入数据来自动学习到子词表，从而避免人工词汇表的限制。

**训练**：通过对给定的语料以BPE进行训练，学习出一个合适的词汇表。训练时，SPM会通过最大化数据的压缩率来选择词汇表。

**分词**：对于输入的句子，SentencePiece模型会根据学到的子词词汇表将句子分解为子词单元。如果某个词汇不在字典中，SPM会将其分解为更小的子词单元。这样，即使遇到未登录词，SPM也有可能能够将其拆解成已知的子词单元进行处理。

- **避免未覆盖词汇问题**：通过子词的构建，SPM能够有效处理未登录词，即使是未在训练数据中出现的词汇，也能通过子词单元进行处理。
- **平衡词汇表大小**：相比传统的基于词汇的分词方法，SPM能够有效地控制词汇表大小，同时保留足够的细节信息。

## 模型构建与实现

在本项目中，尝试了两种神经网络架构来实现机器翻译任务：**基于LSTM的encoder-decoder模型**和**基于注意力机制的Transformer模型**。项目最终选择了Transformer模型进行训练，并完成了基于WMT18数据集的训练任务。

### 1. 基于LSTM的Encoder-Decoder模型

LSTM是一种特别设计的递归神经网络（RNN），它能有效地克服传统RNN在处理长序列时的梯度消失和梯度爆炸问题。LSTM通过引入三个门控机制（输入门、遗忘门和输出门），允许网络保留长期依赖信息，同时抑制无关信息的影响。

在翻译任务中，LSTM用于构建一个**encoder-decoder**架构：

- **Encoder**：接收输入序列，将其逐步编码为一个固定长度的上下文向量（通常为LSTM的最后一个隐藏状态），这个上下文向量被传递给decoder。
- **Decoder**：根据encoder传递的上下文向量，逐步生成目标序列。


#### LSTM模型的训练挑战

理论上LSTM在处理序列任务时能够有较好的表现，但对于翻译任务此类大规模训练任务，在训练模型方面存在障碍。

- LSTM模型由于其递归结构，序列的每个时刻必须依赖前一个时刻的计算结果，这使得LSTM必须逐步地进行decode训练，并行能力差、显存占用高。
- 在现有硬件资源和训练时间条件下，无法进行足够规模的训练。

### 2. 基于Transformer模型

与LSTM相比，**Transformer模型**引入了**自注意力机制（Self-Attention）**，消除了对序列顺序的依赖，完全基于并行计算。Transformer使用编码器和解码器堆叠的结构，在每一层中都采用了自注意力机制和前馈神经网络。Transformer的关键点在于，它通过自注意力机制让模型在处理序列时，可以同时关注输入序列中所有位置的信息，而不仅仅依赖于相邻位置的词汇。

#### Transformer模型的优势

- **并行化**：Transformer模型的计算没有时间步之间的依赖，因此能够在计算时进行高度并行化，能够在更短的时间内完成更大规模的训练任务。
- **长程依赖建模能力强**：通过自注意力机制，Transformer能够有效建模长程依赖，解决了传统RNN和LSTM在处理长序列时的存在的逐步计算时反向传播导致的梯度爆炸和梯度消失的问题。

## 训练与数据集

在训练过程中，我们使用了**WMT18新闻评论数据集**（约250K对平行语料），该数据集包含了英语和多种语言之间的翻译对，主要覆盖了新闻、科技、文化几个领域。考虑到该数据集中话题的偏倚，部分领域的语料可能相对不足，额外使用了**ChatGPT API**生成了30K日常相关的语料，以平衡数据集中的话题分布。

### 训练环境

ubuntu 20.04, Tesla V100-SXM2-32GB, CUDA 12.2,

python==3.8,torch==2.4.1+cu121,torchaudio==2.4.1+cu121,torchvision==0.19.1+cu121
