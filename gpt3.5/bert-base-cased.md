# Model Card for bert-base-cased

The model bert-base-cased is a pre-trained language representation model called BERT (Bidirectional Encoder Representations from Transformers) that achieves state-of-the-art performance on various sentence-level and token-level tasks by using masked language models and bidirectional pre-training.

## Model Details

### Model Description

Model Name: bert-base-cased

Model Architecture: BERT's model architecture is a multi-layer bidirectional Transformer encoder based on the original implementation described in Vaswani et al. (2017).

Training Procedures: The model is pretrained using a "masked language model" (MLM) pre-training objective. This involves randomly masking some of the tokens from the input and predicting the original vocabulary id of the masked word based on its context. Additionally, the model is trained for a binarized next sentence prediction task to understand sentence relationships.

Parameters: The specific number of layers, hidden units, and other hyperparameters of the bert-base-cased model are not mentioned in the provided references. [More Information Needed]

Important Disclaimers: There are no specific disclaimers mentioned in the provided references for the bert-base-cased model. [More Information Needed]

- **Developed by:** Jacob Devlin; Ming-Wei Chang; Kenton Lee; Kristina Toutanova
- **Funded by:** The information provided does not directly mention the people or organizations that fund the bert-base-cased project. Therefore, we need more information to answer this question accurately.
- **Shared by:** The contributors who made the model bert-base-cased available online as a GitHub repo are not mentioned in the provided references. [More Information Needed]
- **Model type:** The model bert-base-cased is a bidirectional encoder representation from transformers (BERT) model, trained using unsupervised training on a plain text corpus, making it a pre-trained deep bidirectional representation model for natural language processing (NLP) tasks. It supports both single sentence and pair of sentences input representations.
- **Language(s):** The model bert-base-cased processes natural human language in a bidirectional manner using a masked language model (MLM) pre-training objective and deep bidirectional representations.
- **License:** The model bert-base-cased is released under the Apache 2.0 license. You can find more information about the license in the `LICENSE` file.
- **Finetuned from model:** The model bert-base-cased is not fine-tuned from another model.
### Model Sources

- **Repository:** https://github.com/google-research/bert
- **Paper:** https://arxiv.org/pdf/1810.04805.pdf
- **Demo:** The link to the demo of the model bert-base-cased is not provided in the given references. [More Information Needed]
## Uses

### Direct Use

The model `bert-base-cased` can be used without fine-tuning, post-processing, or plugging into a pipeline by using the provided script `extract_features.py`. This script allows you to extract contextual embeddings for each input token from the hidden layers of the pre-trained model.

Here is an example of how to use the `extract_features.py` script:

```python
python extract_features.py \
  --input_file=input.txt \
  --output_file=output.jsonl \
  --vocab_file=bert-base-cased-vocab.txt \
  --bert_config_file=bert-base-cased-config.json \
  --init_checkpoint=bert-base-cased-model.ckpt \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=128 \
  --batch_size=8
```

In this example, you need to provide an input file (`input.txt`) containing the text you want to extract contextual embeddings from. The output will be written to an output file (`output.jsonl`) in the JSONL format.

You also need to specify the path to the vocabulary file (`bert-base-cased-vocab.txt`), the BERT configuration file (`bert-base-cased-config.json`), and the initial checkpoint file (`bert-base-cased-model.ckpt`).

You can specify which layers of the model to extract features from using the `--layers` option. In the example above, we are extracting features from the last four layers (`-1,-2,-3,-4`).

Other options such as `max_seq_length` and `batch_size` can be adjusted according to your specific needs.

Note that the `extract_features.py` script assumes that you have the necessary BERT files (vocabulary, configuration, and checkpoint) available.

This code snippet provides a straightforward way to use the `bert-base-cased` model for extracting contextual embeddings without the need for fine-tuning, post-processing, or plugging into a pipeline.

### Downstream Use

The bert-base-cased model can be used when fine-tuned for a specific task or when plugged into a larger ecosystem or app. 

When fine-tuned for a task, the model can be used by adding a simple classification layer on top of the pre-trained model and jointly fine-tuning all the parameters on labeled data from the downstream task. This approach is referred to as the "fine-tuning approach" and is commonly used with BERT. 

To use the bert-base-cased model in a larger ecosystem or app, the model can be plugged in by swapping out the appropriate inputs and outputs. This is made possible by the self-attention mechanism in the Transformer architecture, which allows BERT to model many downstream tasks involving single text or text pairs. For applications involving text pairs, the text pairs can be independently encoded before applying bidirectional cross attention. BERT, however, unifies these stages using self-attention, effectively including bidirectional cross attention between two sentences.

Here is a code snippet example for fine-tuning the bert-base-cased model for a classification task using the Hugging Face `transformers` library:

```python
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch

# Load the pre-trained model and tokenizer
model_name = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Fine-tune the model on labeled data
train_dataset = ...  # Your labeled training dataset
train_dataloader = ...  # Create a data loader for the training dataset

optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
for epoch in range(num_epochs):
    for batch in train_dataloader:
        inputs = tokenizer(batch['text'], padding=True, truncation=True, return_tensors='pt')
        labels = batch['labels']

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

Please note that the code snippet above assumes you have a labeled training dataset and a data loader set up for the training data. You would need to replace `...` with your own code to create these components.

Keep in mind that this code snippet is a simplified example and may need to be adapted to your specific use case.

### Out-of-Scope Use

The model bert-base-cased may foreseeably be misused in a few ways. First, it is important to note that BERT is a language representation model trained on large corpora of text data, including Wikipedia. Therefore, using the model to generate or disseminate false or misleading information would be a misuse. Additionally, the model should not be used to promote hate speech, discrimination, or any form of harmful content.

Moreover, since BERT is a powerful language understanding model, it has the potential to be used for automated content creation, such as generating fake news articles or spam. Users should avoid employing the model for such purposes and must ensure that any generated content is fact-checked and verified before dissemination.

Lastly, although BERT has achieved state-of-the-art performance on various NLP tasks, it is crucial to understand its limitations. The model does not possess a comprehensive understanding of context, and it may struggle with nuanced language, sarcasm, or ambiguity. Users should not solely rely on the model's outputs without critical analysis and interpretation.

Overall, the model bert-base-cased should be used responsibly, with an emphasis on promoting accuracy, fairness, and ethical considerations in all applications.

### Bias, Risks, and Limitations

The known or foreseeable issues stemming from the bert-base-cased model are as follows:

1. Technical limitations: The model is a deep bidirectional architecture, which means it is computationally expensive and requires significant resources for training and inference.

2. Sociotechnical limitations: The model's performance heavily relies on the quality and diversity of the training data. Biases present in the training data can be reflected in the model's output, leading to biased predictions or reinforcing existing societal prejudices.

3. Misunderstandings: The model may generate outputs that appear plausible but are actually incorrect or misleading. It is important to carefully evaluate and validate the model's predictions to avoid spreading misinformation.

4. Foreseeable harms: If the model is deployed in critical applications without proper safeguards, it may have unintended negative consequences. For example, if used for automated decision-making in domains like hiring or lending, biased predictions can perpetuate discrimination.

5. Limited support for certain tasks: The model may not perform optimally in tasks that require understanding of case information, such as Named Entity Recognition or Part-of-Speech tagging, when using the uncased version. In such cases, the cased version should be preferred.

6. Privacy concerns: Although not explicitly mentioned in the references, it is important to note that models like bert-base-cased require large amounts of training data, which may raise privacy concerns if the data contains sensitive or personally identifiable information.

Please note that the above analysis is based on the provided references and may not cover all possible issues. A more comprehensive evaluation would require further analysis and assessment from domain experts.

### Recommendations

Based on the references provided, here are the recommendations regarding the foreseeable issues about the model bert-base-cased:

1. Conduct Ablation Experiments: Perform ablation experiments to better understand the relative importance of different facets of BERT. This will help in identifying any potential weaknesses or areas for improvement in the model.

2. Consider Unidirectional and Shallowly Bidirectional Models: BERT was built upon previous models that are unidirectional or shallowly bidirectional. It is important to note that BERT is a deeply bidirectional system, which means it has a different approach compared to these models. When using BERT, consider the advantages and limitations associated with unidirectional or shallowly bidirectional models.

3. Pre-training on Large Text Corpus: BERT is pre-trained on a large text corpus like Wikipedia to create a general-purpose "language understanding" model. However, if your task involves a large domain-specific corpus (e.g., "movie reviews" or "scientific papers"), additional steps of pre-training on your corpus starting from the BERT checkpoint can be beneficial.

4. Adjust Vocabulary Size: If you plan to use your own vocabulary, make sure to change the `vocab_size` in `bert_config.json`. Failure to update the vocabulary size when using a larger vocabulary may result in NaNs during training on GPU or TPU due to unchecked out-of-bounds access.

Please note that the model card does not explicitly mention any other foreseeable issues. However, it is essential to consider potential ethical concerns, biases, fairness, privacy, and security implications when deploying and using models like bert-base-cased. A thorough analysis by experts in the field of ethics, law, sociology, and rights advocacy is recommended to address these critical aspects.

## Training Details

### Training Data

The training data for the model bert-base-cased consists of the BooksCorpus (800M words) and English Wikipedia (2,500M words) datasets. The Wikipedia dataset is extracted to include only text passages, excluding lists, tables, and headers. For more information on data pre-processing or additional filtering, please refer to the [BERT repository](https://github.com/google-research/bert).

### Training Procedure

#### Preprocessing

The bert-base-cased model follows the pre-training procedure described in the literature on language model pre-training. The model uses a combination of the BooksCorpus (800M words) and English Wikipedia (2,500M words) as the pre-training corpus.

For tokenization, the model uses WordPiece embeddings with a vocabulary size of 30,000 tokens. The first token of every sequence is a special classification token ([CLS]). The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks.

The model differentiates between sentence pairs by separating them with a special token ([SEP]). Additionally, a learned embedding is added to every token to indicate sentence differentiation.

Regarding resizing/rewriting, no specific information is provided in the references for the bert-base-cased model.

In summary, the preprocessing steps for the data in the bert-base-cased model include tokenization using WordPiece embeddings, adding special tokens for classification and sentence differentiation, and using the final hidden state of the [CLS] token as the aggregate sequence representation.

Please note that if you require more specific details about resizing/rewriting or any other aspect of the preprocessing, further information is needed.

#### Training Hyperparameters

The training hyperparameters for the model bert-base-cased are as follows:

- Batch size: The optimal batch size is task-specific, but a range of possible values work well across all tasks. However, the exact value is not specified in the references. [More Information Needed]

- Learning rate: The learning rate is not specified in the references. [More Information Needed]

- Number of training epochs: The number of training epochs is not specified in the references. [More Information Needed]

- Dropout probability: The dropout probability is always kept at 0.1.

#### Speeds, Sizes, Times

BERT-base-cased is a pre-trained language representation model that has been fine-tuned on various NLP tasks. It is trained using a large text corpus consisting of the BooksCorpus (800M words) and English Wikipedia (2,500M words) [6].

Throughput, start or end time, and checkpoint sizes are not explicitly mentioned in the provided references. Therefore, more information is needed to answer this specific question about bert-base-cased.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model bert-base-cased has been evaluated on the General Language Understanding Evaluation (GLUE) benchmark and the Stanford Question Answering Dataset (SQuAD v1.1). 

GLUE benchmark (Wang et al., 2018a) is a collection of diverse natural language understanding tasks. More information about the GLUE datasets can be found in Appendix B.1.

The SQuAD v1.1 is a collection of 100k crowdsourced question/answer pairs (Rajpurkar et al., 2016). The task in SQuAD v1.1 is to predict the answer text span in a given passage from Wikipedia.

Please note that the model bert-base-cased has been found to be outperformed by bert-large-cased across all tasks, especially those with very little training data. The effect of model size on performance is explored in more detail in Section 5.2.

#### Factors

The foreseeable characteristics that will influence how the model bert-base-cased behaves include:

1. Domain and Context: The model's performance will vary depending on the specific domain and context of the text it is applied to. The model has been trained on a large text corpus, including Wikipedia, which provides a general understanding of language. However, its effectiveness may be reduced when dealing with domain-specific or specialized texts.

2. Population Subgroups: The model's performance may also vary across different population subgroups. Evaluation of the model should ideally be disaggregated across factors such as age, gender, ethnicity, and language proficiency to uncover potential disparities in performance. This is crucial to ensure that the model does not exhibit bias or discriminatory behavior towards specific subgroups.

It is important to note that the references provided do not directly address the evaluation of the model bert-base-cased across different factors. Therefore, more information is needed to assess the specific disparities in performance based on different characteristics.

#### Metrics

The metrics used for evaluation in light of tradeoffs between different errors for the model bert-base-cased are not explicitly mentioned in the provided references. Therefore, [More Information Needed].

### Results

Based on the given references, the evaluation results of the model bert-base-cased are as follows:

1. BERT BASE and BERT LARGE outperform all systems on all tasks by a substantial margin, obtaining 4.5% and 7.0% respective average accuracy improvement over the prior state of the art. [Reference 3]

2. Our best performing system outperforms the top leaderboard system by +1.5 F1 in ensembling and +1.3 F1 as a single system. In fact, our single BERT model outperforms the top ensemble system in terms of F1 score. [Reference 2]

3. BERT LARGE significantly outperforms BERT BASE across all tasks, especially those with very little training data. [Reference 1]

Please note that the specific metrics or factors for evaluation are not mentioned in the given references. If you require more detailed evaluation results, please provide additional information or references.

#### Summary

The evaluation results of the model bert-base-cased are as follows:

1. BERT LARGE significantly outperforms BERT BASE across all tasks, especially those with very little training data. [1]
2. BERT BASE and BERT LARGE outperform all systems on all tasks by a substantial margin, with respective average accuracy improvements of 4.5% and 7.0% over the prior state of the art. For the GLUE task MNLI, BERT achieves a 4.6% absolute accuracy improvement. On the official GLUE leaderboard, BERT LARGE obtains a score of 80.5. [2]
3. The best performing BERT system outperforms the top leaderboard system by +1.5 F1 in ensembling and +1.3 F1 as a single system. The single BERT model also outperforms the top ensemble system in terms of F1 score. [3]
4. For the SQuAD v2.0 task, BERT extends the BERT v1.1 model by allowing for the possibility of no short answer in the provided paragraph, resulting in a more realistic problem definition. [4]
5. BERT achieves new state-of-the-art results on eleven NLP tasks, including a GLUE score of 80.5% (7.7% absolute improvement), MultiNLI accuracy of 86.7% (4.6% absolute improvement), SQuAD v1.1 Test F1 of 93.2 (1.5 point absolute improvement), and SQuAD v2.0 Test F1 of [More Information Needed]. [6]
6. BERT can be adapted to various NLP tasks easily, including sentence-level, sentence-pair-level, word-level, and span-level tasks, with minimal task-specific modifications. [7]
7. Pre-trained representations in BERT reduce the need for heavily-engineered task-specific architectures, achieving state-of-the-art performance on sentence-level and token-level tasks, surpassing many task-specific architectures. [8]

Overall, bert-base-cased demonstrates strong performance across multiple NLP tasks and outperforms previous state-of-the-art systems. It is particularly effective in tasks with limited training data.

## Model Examination

The model bert-base-cased is a pre-trained language representation model that achieves state-of-the-art performance on various NLP tasks. It is trained on a large text corpus, such as Wikipedia, using unsupervised and deeply bidirectional methods (Reference 3). This pre-training allows the model to learn general language understanding, which reduces the need for heavily-engineered task-specific architectures (Reference 4).

BERT advances the state of the art for eleven NLP tasks (Reference 4). However, there is no specific information provided in the references about an experimental section on explainability/interpretability for the bert-base-cased model. Therefore, more information is needed in this regard.

## Environmental Impact

- **Hardware Type:** The hardware type that the model bert-base-cased is trained on is not mentioned in the given references. [More Information Needed]
- **Software Type:** The model bert-base-cased is trained on a general-purpose "language understanding" model using the BERT (Bidirectional Encoder Representations from Transformers) method. It is a pre-training language representation model that obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks. The model is trained on a large text corpus, such as Wikipedia, and can be fine-tuned with just one additional output layer to create state-of-the-art models for specific downstream NLP tasks. The software type for the model is [More Information Needed].
- **Hours used:** The amount of time used to train the model bert-base-cased is not mentioned in the provided references. [More Information Needed]
- **Cloud Provider:** The cloud provider on which the model bert-base-cased is trained is not mentioned in the provided references. [More Information Needed]
- **Carbon Emitted:** The amount of carbon emitted when training the model bert-base-cased is not mentioned in the provided references. [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The model architecture of bert-base-cased is a multi-layer bidirectional Transformer encoder, based on the original implementation described in Vaswani et al. (2017) and released in the tensor2tensor library [1]. The model consists of L=12 layers, with a hidden size of H=768 and A=12 self-attention heads. The total number of parameters is 110M [4].

The objective of bert-base-cased is to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers [11]. This is achieved through a "masked language model" (MLM) pre-training objective, where some of the tokens from the input are randomly masked, and the objective is to predict the original vocabulary id of the masked word based on its context [8].

Please note that the given references provide more detailed information about the model architecture and objective of bert-base-cased.

### Compute Infrastructure

The compute infrastructure information about the model bert-base-cased is not provided in the given references. [More Information Needed].

## Citation

```
@misc{jacob-bert,
    author = {Jacob Devlin and
              Ming-Wei Chang and
              Kenton Lee and
              Kristina Toutanova},
    title  = {BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
    url    = {https://arxiv.org/pdf/1810.04805.pdf}
}
```

