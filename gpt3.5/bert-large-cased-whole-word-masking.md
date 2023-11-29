# Model Card for bert-large-cased-whole-word-masking

The model bert-large-cased-whole-word-masking is a 24-layer, 1024-hidden, and 16-heads pre-trained language representation model that utilizes bidirectional pre-training with masked language models to enable deep bidirectional representations, achieving state-of-the-art performance on various sentence-level and token-level natural language processing tasks. It is part of the BERT (Bidirectional Encoder Representations from Transformers) framework, and the pre-trained models can be accessed from the provided link.

## Model Details

### Model Description

Model Name: bert-large-cased-whole-word-masking

Description:
BERT-Large is a language representation model that is designed to pretrain deep bidirectional representations from unlabeled text. It is built upon recent work in pre-training contextual representations. The model architecture is a multi-layer bidirectional Transformer encoder based on the original implementation described in Vaswani et al. (2017). BERT-Large uses a "masked language model" (MLM) pre-training objective, inspired by the Cloze task.

Training Procedures:
To train BERT-Large, a deep bidirectional representation is obtained by randomly masking 15% of the input tokens and predicting those masked tokens. The final hidden vectors corresponding to the masked tokens are fed into an output softmax over the vocabulary. The training data generator chooses 15% of token positions at random for prediction, replacing the chosen tokens with the [MASK] token 80% of the time and a random token 10% of the time.

Parameters:
The model size of BERT-Large is chosen to be the same as OpenAI GPT for comparison purposes. It uses bidirectional self-attention in the Transformer encoder, unlike the GPT Transformer which uses constrained self-attention where every token can only attend to context on its left.

Important Disclaimers:
- BERT-Large's model architecture and training procedures are based on previous work in pre-training contextual representations.
- BERT-Large uses a masked language model (MLM) pre-training objective, which introduces a mismatch between pre-training and fine-tuning as the [MASK] token does not appear during fine-tuning.
- BERT-Large demonstrates the importance of bidirectional pre-training for language representations and reduces the need for task-specific architectures.
- BERT-Large achieves state-of-the-art performance on a large suite of sentence-level and token-level tasks, outperforming many task-specific architectures.

Please note that this is a high-level summary of the model card description for bert-large-cased-whole-word-masking. For more detailed information, please refer to the provided references.

- **Developed by:** Jacob Devlin; Ming-Wei Chang; Kenton Lee; Kristina Toutanova
- **Funded by:** The people or organizations that fund the project of the model bert-large-cased-whole-word-masking are Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.
- **Shared by:** The contributors who made the model bert-large-cased-whole-word-masking available online as a GitHub repo are not mentioned in the provided references. [More Information Needed]
- **Model type:** The model bert-large-cased-whole-word-masking is a pre-trained language representation model that utilizes bidirectional pre-training, unsupervised learning, and text data as its modality.
- **Language(s):** The model bert-large-cased-whole-word-masking processes uncased natural human language.
- **License:** The license used for the model bert-large-cased-whole-word-masking is the Apache 2.0 license. The link to the license is not provided in the given references. [More Information Needed]
- **Finetuned from model:** The base model from which the model bert-large-cased-whole-word-masking is fine-tuned is not mentioned in the provided references. [More Information Needed]
### Model Sources

- **Repository:** https://github.com/google-research/bert
- **Paper:** https://arxiv.org/pdf/1810.04805.pdf
- **Demo:** The demo for the model bert-large-cased-whole-word-masking is not mentioned in the provided references. [More Information Needed]
## Uses

### Direct Use

To use the model bert-large-cased-whole-word-masking without fine-tuning, post-processing, or plugging into a pipeline, you can follow these steps:

1. Install the required libraries:
```
!pip install transformers
```

2. Import the necessary modules:
```python
from transformers import BertTokenizer, BertModel

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-cased-whole-word-masking')

# Load the model
model = BertModel.from_pretrained('bert-large-cased-whole-word-masking')
```

3. Tokenize the input text:
```python
text = "Example sentence to encode."
tokens = tokenizer.tokenize(text)
```

4. Convert the tokens to input IDs:
```python
input_ids = tokenizer.convert_tokens_to_ids(tokens)
```

5. Generate the model embeddings:
```python
# Add a batch dimension
input_ids = torch.tensor([input_ids])

# Forward pass through the model
outputs = model(input_ids)

# Get the embeddings
embeddings = outputs.last_hidden_state
```

Now you have the model embeddings for the input text using the bert-large-cased-whole-word-masking model. Please note that these embeddings are obtained without fine-tuning, post-processing, or plugging into a pipeline.

Please let me know if I can help you with anything else.

### Downstream Use

The bert-large-cased-whole-word-masking model can be used in two steps: pre-training and fine-tuning. During pre-training, the model is trained on unlabeled data over different pre-training tasks. For fine-tuning, the model is initialized with the pre-trained parameters and fine-tuned using labeled data from downstream tasks. The fine-tuning process is straightforward and can be done by swapping out the appropriate inputs and outputs.

Once the model is fine-tuned, it can be used for various tasks by plugging it into a larger ecosystem or app. For tasks involving single text or text pairs, such as paraphrasing, entailment, question answering, or text classification, the input can be sentence A and sentence B from pre-training. For classification tasks, the [CLS] token representation can be fed into an output layer. For token-level tasks, the token representations can be fed into an output layer.

To use the bert-large-cased-whole-word-masking model in a larger ecosystem or app, you can use the Huggingface Transformers library. Here is an example code snippet for text classification using the fine-tuned model:

```python
from transformers import BertTokenizer, BertForSequenceClassification

# Load the fine-tuned model
model = BertForSequenceClassification.from_pretrained('path/to/fine-tuned/model')

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-cased-whole-word-masking')

# Encode the input text
text = "This is an example sentence."
inputs = tokenizer(text, return_tensors='pt')

# Make predictions
outputs = model(**inputs)
predictions = outputs.logits.argmax(dim=1)

# Convert predictions to labels
labels = ['Label 1', 'Label 2', 'Label 3']
predicted_label = labels[predictions.item()]

print("Predicted label:", predicted_label)
```

Please note that the code snippet is just a general example and may need to be modified based on your specific task and data.

### Out-of-Scope Use

The model bert-large-cased-whole-word-masking is designed to pretrain deep bidirectional representations from unlabeled text by conditioning on both left and right context in all layers. It can be finetuned with just one additional output layer to create state-of-the-art models for various natural language processing tasks.

However, it is important to consider potential misuse of this model. One foreseeable misuse could be the generation of biased or harmful content. As the model is trained on large amounts of text data, it may inadvertently learn and reproduce biases present in the training data. Users should be cautious and actively mitigate any biases that may arise from the model's outputs.

Additionally, users should not engage in activities that violate privacy or ethical standards. The model should not be used to generate or disseminate false or misleading information, engage in malicious activities such as hacking or phishing, or infringe on intellectual property rights.

It is crucial for users to be responsible and accountable when using the bert-large-cased-whole-word-masking model, ensuring that its outputs are used in a fair, unbiased, and lawful manner.

[More Information Needed]

### Bias, Risks, and Limitations

Based on the references provided, the known or foreseeable issues stemming from the model bert-large-cased-whole-word-masking are as follows:

1. Technical Limitation: The model relies on pre-training using a "masked language model" (MLM) objective, where some tokens from the input are randomly masked. This may lead to the model being biased towards the specific type of language patterns found in the training data and may not generalize well to other types of text.

2. Sociotechnical Limitation: The model's performance is heavily dependent on the quality and representativeness of the training data. If the training data is biased or lacks diversity, the model may learn and perpetuate those biases, leading to biased predictions and potential harms when used in real-world applications.

3. Misunderstandings: The model may not fully capture the semantic meaning of certain words or phrases, leading to potential misunderstandings when interpreting the model's predictions. Additionally, the model may struggle with understanding the context and nuance of ambiguous statements, resulting in misinterpretations.

4. Foreseeable Harms: The model's large size and computational requirements may limit its accessibility and usability for users with limited resources. Furthermore, if deployed without proper safeguards and monitoring, the model could be used for malicious purposes such as generating deceptive or harmful content.

5. Ethical Concerns: The use of pre-trained models like bert-large-cased-whole-word-masking may contribute to the concentration of power and resources in the hands of a few organizations or researchers who have access to large-scale computing infrastructure. This could further exacerbate existing inequalities in the field of natural language processing and limit the participation of smaller research groups or organizations.

It is important to note that these issues may not be exhaustive, and further research and analysis are necessary to fully understand the potential risks and limitations associated with this model.

### Recommendations

The bert-large-cased-whole-word-masking model is a 24-layer, 1024-hidden, 16-heads model with 340M parameters. It is a variant of BERT-Large that includes whole word masking.

Whole word masking is enabled during data generation by passing the flag `--do_whole_word_mask=True` to `create_pretraining_data.py`.

There are no specific recommendations provided in the references regarding foreseeable issues with the bert-large-cased-whole-word-masking model. However, it is important to note that when using these models, it is advised to make it clear in the paper that the Whole Word Masking variant of BERT-Large is being used.

It is also important to consider potential issues related to the ethical and societal implications of using language models like bert-large-cased-whole-word-masking. As a sociotechnic, it is crucial to engage with lawyers, ethicists, sociologists, or rights advocates to assess and address any potential biases, fairness concerns, or ethical considerations that may arise from the model's deployment and use.

In summary, the references do not provide specific recommendations about foreseeable issues with the bert-large-cased-whole-word-masking model. However, it is essential to consider the ethical and societal implications and engage with relevant experts to ensure responsible and unbiased use of the model.

## Training Details

### Training Data

The training data for the model bert-large-cased-whole-word-masking consists of a combination of the BooksCorpus (800M words) and English Wikipedia (2,500M words) datasets. The data pre-processing and cleanup steps for extracting a usable corpus are not mentioned in the references. However, it is important to note that the model uses a document-level corpus rather than a shuffled sentence-level corpus to extract long contiguous sequences. For more information, you can refer to the [reference 2](http://commoncrawl.org/) and the [BERT paper](https://arxiv.org/abs/1810.04805).

### Training Procedure

#### Preprocessing

The preprocessing for the data of the model bert-large-cased-whole-word-masking involves tokenization using the WordPiece method with a 30,000 token vocabulary. Each sequence starts with a special classification token ([CLS]), and sentence pairs are packed together into a single sequence. The sentences are differentiated by a special token ([SEP]). Additionally, a learned embedding is added to every token indicating sentence separation. 

For the Whole Word Masking technique, all tokens corresponding to a word are masked at once. The masking rate remains the same, and the training involves predicting each masked WordPiece token independently.

To enable Whole Word Masking during data generation, the flag `--do_whole_word_mask=True` should be passed to `create_pretraining_data.py`.

[More Information Needed]

#### Training Hyperparameters

The training hyperparameters for the model bert-large-cased-whole-word-masking are not explicitly mentioned in the provided references. Therefore, we need more information to provide the specific training hyperparameters for this model.

#### Speeds, Sizes, Times

The model bert-large-cased-whole-word-masking is a variant of BERT-Large that uses whole word masking during pre-training. It has the following specifications:

- Number of layers: 24
- Hidden size: 1024
- Number of attention heads: 16
- Number of parameters: 340M

Regarding the detail throughput, start or end time, and checkpoint sizes for the model bert-large-cased-whole-word-masking, there is no specific information provided in the references. Therefore, we would need more information to answer these questions.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model bert-large-cased-whole-word-masking has been evaluated on the following benchmarks or datasets:

1. General Language Understanding Evaluation (GLUE) benchmark: The model has been tested on GLUE, which is a collection of diverse natural language understanding tasks (Wang et al., 2018a). Detailed descriptions of GLUE datasets can be found in Appendix B.1.

2. Stanford Question Answering Dataset (SQuAD v1.1): The model has been evaluated on SQuAD v1.1, which is a collection of 100k crowdsourced question/answer pairs (Rajpurkar et al., 2016). The task is to predict the answer text span in a given passage from Wikipedia.

3. CoNLL-2003 Named Entity Recognition (NER) task: The model has been applied to the CoNLL-2003 NER task (Tjong Kim Sang and De Meulder, 2003). The input to BERT uses a case-preserving WordPiece model, and the maximal document context provided by the data is included. The task is formulated as a tagging task, where the representation of the first sub-token is used as the input to the token-level classifier over the NER label set.

Please note that the model card description may need more information or specific details regarding these benchmarks or datasets.

#### Factors

The foreseeable characteristics that will influence how the model bert-large-cased-whole-word-masking behaves include domain, context, and population subgroups. 

1. Domain: The model's performance will vary depending on the specific domain of the text it is applied to. The model is trained on a diverse range of text, but it may have limitations in specialized domains where the training data is limited or significantly different.

2. Context: The model's behavior will be influenced by the context in which it is used. Different contexts may introduce biases or limitations that affect the model's performance. For example, if the model is used in a context with specific slang or jargon, its performance may be impacted.

3. Population Subgroups: The model's performance may vary across population subgroups. It is essential to evaluate the model's performance across different demographic groups to uncover any disparities or biases. Disaggregating the evaluation results can help identify potential issues and ensure fairness and inclusivity.

To evaluate the model's performance across these factors, it is necessary to conduct specific analysis and evaluation. This may involve testing the model on domain-specific datasets, examining its behavior in different contexts, and analyzing its performance across different population subgroups. Such evaluations can help uncover any disparities or biases and guide improvements to enhance the model's performance and fairness.

#### Metrics

The metrics used for evaluation in light of tradeoffs between different errors for the model bert-large-cased-whole-word-masking are not explicitly mentioned in the given references. [More Information Needed]

### Results

The evaluation results of the model bert-large-cased-whole-word-masking are not provided in the given references. [More Information Needed]

#### Summary

The evaluation results for the model bert-large-cased-whole-word-masking are as follows:

- BERT LARGE outperforms BERT BASE across all tasks, especially those with little training data.
- BERT LARGE and BERT BASE outperform all other systems on all tasks by a significant margin.
- BERT LARGE obtains a score of 80.5 on the official GLUE leaderboard.
- In the question answering task, BERT uses a packed sequence representation and introduces start and end vectors during fine-tuning.
- BERT achieves state-of-the-art results on eleven NLP tasks, including question answering and language inference.
- BERT reduces the need for task-specific architectures and outperforms many task-specific models.
- BERT can be easily adapted to different types of NLP tasks with minimal modifications.

Please note that the evaluation results provided are based on the references provided and may not cover all aspects of the model's performance. For more detailed information, further evaluation and analysis would be required.

## Model Examination

The model bert-large-cased-whole-word-masking is a variant of BERT-Large that incorporates Whole Word Masking. Whole Word Masking is a masking strategy used during pre-training with the masked language model (MLM) objective of BERT. It aims to reduce the mismatch between pre-training and fine-tuning by masking entire words instead of individual tokens.

The bert-large-cased-whole-word-masking model has the same structure and vocabulary as the original BERT-Large model. It is trained on a large text corpus, such as Wikipedia, to learn a general-purpose "language understanding" model. This pre-trained model can then be fine-tuned for downstream natural language processing (NLP) tasks.

Unfortunately, the specific details about the experimental section on explainability/interpretability for the bert-large-cased-whole-word-masking model are not provided in the given references. [More Information Needed]

## Environmental Impact

- **Hardware Type:** The model bert-large-cased-whole-word-masking is trained on a large corpus using a GPU or TPU.
- **Software Type:** The model bert-large-cased-whole-word-masking is trained on the software type of "pre-training language representations" using a large text corpus such as BooksCorpus and English Wikipedia.
- **Hours used:** The training time for the model bert-large-cased-whole-word-masking is not explicitly mentioned in the provided references. Therefore, I require more information to answer this question.
- **Cloud Provider:** The cloud provider on which the model bert-large-cased-whole-word-masking is trained is not explicitly mentioned in the provided references. Therefore, more information is needed to determine the cloud provider used for training this model.
- **Carbon Emitted:** The amount of carbon emitted when training the model bert-large-cased-whole-word-masking is not provided in the given references. [More Information Needed]
## Technical Specification

### Model Architecture and Objective

Model Architecture:
The model architecture of bert-large-cased-whole-word-masking is a multi-layer bidirectional Transformer encoder. It is based on the original implementation described in Vaswani et al. (2017) and released in the tensor2tensor library. The implementation is almost identical to the original Transformer model, so a detailed background description is omitted. For more information, please refer to Vaswani et al. (2017) and "The Annotated Transformer" guide.

Objective:
The objective of bert-large-cased-whole-word-masking is to pre-train contextual representations using a "masked language model" (MLM) pre-training objective. This objective is inspired by the Cloze task and helps alleviate the unidirectionality constraint. In the MLM objective, some tokens from the input are randomly masked, and the objective is to predict the original vocabulary id of the masked word based only on its context. By using bidirectional self-attention, the model is able to fuse the left and right context, enabling the pretraining of a deep bidirectional Transformer.

### Compute Infrastructure

The compute infrastructure information about the model bert-large-cased-whole-word-masking is not provided in the given references. [More Information Needed]

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

