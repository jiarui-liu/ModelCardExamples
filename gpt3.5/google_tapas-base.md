# Model Card for google/tapas-base

The model google/tapas-base is a table parser that performs question answering over tables without generating logical forms. It extends BERT's architecture and is trained using weak supervision, predicting a minimal program by selecting table cells and applying an aggregation operation. The model achieves better accuracy compared to prior attempts and is pre-trained on millions of tables and related text segments from Wikipedia.

## Model Details

### Model Description

Model Name: google/tapas-base

Model Architecture:
The architecture of google/tapas-base is based on BERT's encoder with additional positional embeddings used to encode tabular structure. The table is flattened into a sequence of words, and the words are split into word pieces (tokens). The question tokens are concatenated with the table tokens. Two classification layers are added for selecting table cells and aggregation operators. The model also introduces a special embedding to mark whether a cell token was the answer to the previous question.

Training Procedures:
To pre-train TAPAS, a large number of tables from Wikipedia are used. The model learns correlations between text and tables, as well as between the cells of a column and their header. The pre-training is done using a masked language model objective, where tokens from the text segment and the table are masked, and the objective is to predict the original masked token based on the context.

Parameters:
[More Information Needed]

Important Disclaimers:
[More Information Needed]

- **Developed by:** Jonathan Herzig; Paweł Krzysztof Nowak; Thomas Müller; Francesco Piccinno; Julian Martin Eisenschlos
- **Funded by:** The people or organizations that fund the project of the model google/tapas-base are:

1. Google PhD fellowship
2. [More Information Needed]
- **Shared by:** The contributors who made the model google/tapas-base available online as a GitHub repo are Jonathan Herzig, Paweł Krzysztof Nowak, Thomas Müller, Francesco Piccinno, and Julian Martin Eisenschlos.
- **Model type:** The model google/tapas-base is a weakly supervised question answering model that reasons over tables without generating logical forms, trained through pre-training on a large number of tables from Wikipedia and fine-tuning on semantic parsing datasets.
- **Language(s):** The model google/tapas-base uses natural human language in the form of textual and tabular data to reason over tables and answer questions without generating logical forms.
- **License:** The name and link to the license being used for the model google/tapas-base is not provided in the given references. [More Information Needed]
- **Finetuned from model:** The base model for google/tapas-base is not mentioned in the given references. [More Information Needed]
### Model Sources

- **Repository:** https://github.com/google-research/tapas
- **Paper:** https://arxiv.org/pdf/2004.02349.pdf
- **Demo:** The model card for google/tapas-base:

## Model Details

- Model Name: google/tapas-base
- Model Class: Table Question Answering
- Model Architecture: Transformer-based
- Model Training Code: [Code and checkpoints](https://github.com/google-research/tapas) for training the transformer-based Table QA models introduced in the paper [TAPAS: Weakly Supervised Table Parsing via Pre-training](#how-to-cite-tapas).
- Pre-trained Model Availability: The pre-trained model is publicly available at [https://github.com/google-research/tapas](https://github.com/google-research/tapas).
- Model Card Updates Contact: [Your Name](mailto:your-email@example.com)

## Intended Use

The model google/tapas-base is intended to be used for table question answering tasks. It is designed to parse and understand tabular data and provide accurate answers to queries about the table content. The model is well-suited for tasks that involve extracting information from structured tabular data.

## Training Data

The model is trained using weak supervision and pre-training techniques described in the paper [TAPAS: Weakly Supervised Table Parsing via Pre-training](#how-to-cite-tapas). The training data is not explicitly mentioned in the provided references.

## Evaluation Data

The evaluation data used to assess the performance of the model is not specified in the provided references.

## Model Limitations

The limitations and potential biases of the model are not discussed in the provided references.

## Ethical Considerations

The ethical considerations related to the model google/tapas-base are not discussed in the provided references.

## Caveats and Recommendations

The caveats and recommendations for using the model google/tapas-base are not mentioned in the provided references.

## Demo

A demo of the model google/tapas-base is not specified in the provided references. [More Information Needed]
## Uses

### Direct Use

To use the model google/tapas-base without fine-tuning, post-processing, or plugging into a pipeline, you can directly pass in a table and a question as inputs to the model for inference. The model will then predict a minimal program by selecting a subset of the table cells and a possible aggregation operation to be executed on top of them.

Here's a code snippet demonstrating how to use the model for inference:

```python
from transformers import TapasTokenizer, TapasForQuestionAnswering

# Load the tokenizer and model
tokenizer = TapasTokenizer.from_pretrained("google/tapas-base")
model = TapasForQuestionAnswering.from_pretrained("google/tapas-base")

# Define the table and question
table = [
    ["City", "Country", "Population"],
    ["New York", "USA", "8.4 million"],
    ["London", "UK", "8.9 million"],
    ["Tokyo", "Japan", "14 million"],
]
question = "What is the population of New York?"

# Tokenize the table and question
inputs = tokenizer(table=table, queries=question, padding="max_length", truncation=True, return_tensors="pt")

# Perform inference
outputs = model(**inputs)

# Get the predicted answer
answer = outputs.answer
print("Predicted Answer:", answer)
```

Please note that this code snippet assumes you have the `transformers` library installed. You can install it via `pip install transformers`.

By using the above code, you can directly obtain the predicted answer from the model without any additional steps.

Please let me know if you need further assistance!

### Downstream Use

The google/tapas-base model can be used when fine-tuned for a task or when plugged into a larger ecosystem or app. This model is specifically designed for table parsing tasks and can be fine-tuned for various downstream tasks such as table entailment or question answering on tables.

To use the google/tapas-base model for fine-tuning, you can follow the standard procedure of fine-tuning a transformer-based model. Here is an example code snippet using the Huggingface Transformers library:

```python
from transformers import TapasTokenizer, TapasForQuestionAnswering

tokenizer = TapasTokenizer.from_pretrained('google/tapas-base')
model = TapasForQuestionAnswering.from_pretrained('google/tapas-base')

# Fine-tuning code goes here

```

Please note that the code snippet provided is just an example and you will need to adapt it to your specific use case and dataset.

For more information on how to fine-tune and use the google/tapas-base model, you can refer to the official Tapas GitHub repository (https://github.com/google-research/tapas) and the associated documentation provided there.

[More Information Needed]

### Out-of-Scope Use

The model google/tapas-base is a powerful tool for question answering over tables. However, it is important to consider potential misuse and guide users on what they ought not to do with the model.

Given the provided references, it is clear that the model is designed for weakly supervised question answering and aims to avoid generating logical forms. It predicts a minimal program by selecting a subset of table cells and an aggregation operation to be executed on top of them. This approach allows TAPAS to learn operations from natural language without the need for a formalism.

While the model has many potential applications, it is crucial to highlight some foreseeable misuses and caution users about them. Here are a few points to consider:

1. Inappropriate data usage: Users should not use the model to process or analyze sensitive or private data without appropriate consent or authorization. The model is a tool and should be used responsibly, respecting privacy and legal considerations.

2. Unfair bias amplification: The model may learn biases present in the training data and inadvertently amplify them. Users should be cautious when applying the model in scenarios where fairness and equity are critical, such as hiring decisions or criminal justice. Careful evaluation and bias mitigation strategies should be employed to ensure fair and just outcomes.

3. Misinterpretation of results: The model may provide answers that are technically correct but misunderstood or misinterpreted by the user. It is essential to educate users about the limitations of the model and encourage critical thinking when interpreting the results. The model should not be solely relied upon for making important decisions without human judgment and verification.

4. Failure in complex or out-of-scope queries: The model may struggle with complex or out-of-scope queries that go beyond its predefined capabilities. Users should be aware of the model's limitations and avoid relying on it for tasks that require deep domain expertise or complex reasoning beyond the scope of the model's training data.

It is important to provide clear documentation and guidelines to users about these potential misuses and limitations. Additionally, continuous monitoring and feedback from users can help identify and address any unforeseen issues that may arise during the usage of the model.

### Bias, Risks, and Limitations

The model google/tapas-base, a weakly supervised question answering model that reasons over tables without generating logical forms, has several known or foreseeable issues:

1. In some tables, the WIKISQL dataset stores "REAL" numbers as "TEXT" format, leading to incorrect results for comparison and aggregation examples. This penalizes systems that perform their own execution, rather than relying on the SQL query execution used by WIKISQL. This issue can affect the accuracy of the model's execution on tables with such discrepancies. [1]

2. The model's accuracy for predicting numbers is evaluated using a soft metric that relaxes the strict accuracy measure. While the model performs well in guessing numbers close to the target, this soft metric may not capture the complete accuracy of the model. [5]

3. The model currently focuses on tables with single contexts and may face limitations when representing databases with multiple tables as context. Future work aims to extend the model to effectively handle larger tables and multiple tables as context. [3]

4. While TAPAS achieves better or competitive results compared to state-of-the-art semantic parsers, there may still be limitations and misunderstandings in the model's understanding and interpretation of complex queries and table structures. [2]

5. The model's training heavily relies on weak supervision and large-scale data of text-table pairs. This may introduce biases and limitations in the dataset that can affect the model's generalization and performance on real-world scenarios. [2]

6. The model's performance and accuracy depend on the quality and relevance of the data it was pre-trained on, which includes tables and text segments crawled from Wikipedia. The limitations and biases present in the Wikipedia data can impact the model's performance and understanding of real-world tables. [10]

7. The model's ability to predict the denotation relies on selecting table cells and applying aggregation operators. While this approach allows for learning operations from natural language, without the need for a formalism, it may still have limitations in capturing complex queries and reasoning over tables with nuanced structures. [11]

Overall, the model google/tapas-base has made significant advancements in table-based question answering but still has limitations in handling certain table formats, predicting numbers accurately, understanding complex queries and table structures, and generalizing to real-world scenarios. Further research and improvements are needed to address these issues and enhance the model's performance.

### Recommendations

Based on the provided references, here are the recommendations with respect to the foreseeable issues about the model google/tapas-base:

1. Annotation by non-experts: As the model can be supervised by non-experts, it is recommended to ensure clear guidelines and instructions for annotators to prevent potential biases or inaccuracies in the annotations.

2. Handling large tables: The model may struggle with tables that are too big to fit in 512 tokens. To address this, it is recommended to explore techniques such as table compression or filtering to encode only relevant content from large tables.

3. Penalties for selecting no cells: The model sometimes selects no cells as the answer, which may not be desired behavior. It is recommended to introduce penalties or adjustments in the training process to discourage this behavior and encourage the model to select relevant cells.

4. Limitations with scalar differences: The model may not be capable of handling cases where the answer involves the difference between scalars. It is recommended to identify such cases during annotation and handle them separately to avoid inaccuracies in the model's predictions.

5. Unclassified cases: A significant percentage of cases could not be classified to a particular phenomenon. It is recommended to further investigate these cases to understand the reasons behind the lack of classification and potentially improve the model's performance in identifying different phenomena.

6. Computational requirements: Training the large model with 512 sequence length requires a TPU. To make the model trainable on GPUs, options such as reducing the sequence length or batch size can be explored, keeping in mind that these changes may affect the model's accuracy.

7. Limitations with multiple tables or large databases: The model currently only handles single tables that can fit in memory. It is recommended to consider future work on addressing the limitations of handling very large tables or databases with multiple tables, such as compressing or filtering the content.

Overall, it is important to ensure that the model is used with caution, considering its limitations and potential biases that may arise during annotation or training. Ongoing monitoring and evaluation should be conducted to identify and address any unforeseen issues that may arise in real-world applications.

## Training Details

### Training Data

The training data for the model google/tapas-base consists of semantic parsing datasets such as WIKITQ, SQA, and WIKISQL. The dataset includes complex questions on Wikipedia tables, decomposed questions from WIKITQ, and templates paraphrased by crowd workers. The data is pre-processed by converting denotations to question, cell coordinates, scalar answer triples, and filling scalar answer information based on the denotation. [More Information Needed]

### Training Procedure

#### Preprocessing

The data preprocessing for the google/tapas-base model involves tokenization, resizing, and rewriting. Here are the details:

1. Tokenization: The standard BERT tokenizer is applied on questions, table cells, and headers using a shared vocabulary of 32k word pieces. Numbers and dates are parsed similarly to the Neural Programmer (Neelakantan et al., 2017).

2. Resizing/Rewriting: To ensure efficient pre-training, a word piece sequence length budget is imposed (e.g., 128 in the final experiments). This means that the combined length of the tokenized text and table cells must fit within this budget. To achieve this, a snippet of 8 to 16 word pieces from the associated text is randomly selected. For the table, only the first word of each column name and cell is initially added. Words are then added turn-wise until the word piece budget is reached. This resizing process is performed for each table.

In summary, the data is tokenized using the BERT tokenizer and resized/rewritten to fit within a word piece sequence length budget.

#### Training Hyperparameters

The training hyperparameters for the model google/tapas-base are not explicitly mentioned in the given references. Therefore, [More Information Needed] to provide the detail training hyperparameters for the model google/tapas-base.

#### Speeds, Sizes, Times

The model card for google/tapas-base provides information about the model's performance, training process, and available resources. However, it does not explicitly mention details such as throughput, start or end time, or checkpoint sizes. Therefore, we would need more information to provide an answer to this question.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model google/tapas-base evaluates on the following benchmarks and datasets:

1. WIKITQ: This dataset consists of complex questions on Wikipedia tables. The model predicts cell selection or scalar answers for this dataset.

2. SQA: This dataset was constructed by decomposing a subset of highly compositional questions from WIKITQ. The model predicts cell selection or scalar answers for this dataset as well.

3. WIKISQL: This dataset focuses on translating text to SQL. The model generates the reference answer, aggregation operator, and cell coordinates from the reference SQL provided using its own SQL implementation running on the JSON tables.

The model's performance is evaluated using denotation accuracy on WIKITQ and SQA datasets. For WIKISQL, the model's accuracies are reported against its reference answers.

[More Information Needed]

#### Factors

The foreseeable characteristics that will influence how the model google/tapas-base behaves include:

1. Domain and Context: The model is specifically designed for semantic parsing tasks that require discrete reasoning over tables. It is trained on tables and text segments crawled from Wikipedia, which makes it more suitable for tasks involving structured data in a similar domain. It may not perform as well in tasks outside this domain.

2. Population Subgroups: The model's performance may vary across different population subgroups. To uncover potential disparities in performance, evaluation should ideally be disaggregated across factors such as language, cultural context, and data sources. This will help identify any biases or limitations of the model in specific subgroups.

3. Evaluation Disaggregation: In order to thoroughly understand the model's behavior, evaluation should be disaggregated across various factors. This could include analyzing performance differences based on dataset characteristics, question types, table structures, and table sizes. Disaggregating the evaluation will provide insights into the strengths and weaknesses of the model across different scenarios.

In summary, the model's behavior will be influenced by the domain and context it was trained on, as well as potential disparities in performance across different population subgroups. Disaggregating the evaluation across factors will help uncover any biases or limitations of the model in specific scenarios.

#### Metrics

The metrics used for evaluation in the context of tradeoffs between different errors for the model google/tapas-base are not explicitly mentioned in the provided references. [More Information Needed]

### Results

Evaluation results of the model google/tapas-base based on the Factors and Metrics are as follows:

1. For the WIKITQ dataset, the model trained only from the original training data achieves an accuracy of 42.6, surpassing similar approaches (Neelakantan et al., 2015).
2. When the model is pre-trained on WIKISQL or SQA datasets, it achieves accuracies of 48.7 and 48.8, respectively.
3. The model achieves an accuracy of 86.4 on the SQA dataset when provided with full supervision in the form of extracted SQL queries.
4. The model achieves close to state-of-the-art performance on the WIKISQL dataset with an accuracy of 83.6 compared to 83.9 achieved by Min et al. (2019).
5. On the conversational SQA dataset, the model improves the state-of-the-art accuracy from 55.1 to 67.2.
6. TAPAS performs better or on par compared to other semantic parsing and question answering models on different semantic parsing datasets.
7. TAPAS achieves better accuracy compared to prior attempts to reason over tables without generating logical forms, and it handles more question types, including aggregation, and directly handles a conversational setting.
8. Overall, TAPAS achieves better or competitive results compared to state-of-the-art models in question answering over tables.

Please note that more specific metrics or detailed evaluation results are not provided in the references.

#### Summary

The evaluation results for the google/tapas-base model are as follows:

- For the WIKITQ dataset, the model achieves a denotation accuracy of 42.6, which surpasses similar approaches.
- When the model is pre-trained on the WIKISQL dataset, it achieves a denotation accuracy of 48.7. Similarly, when pre-trained on the SQA dataset, it achieves a denotation accuracy of 48.8.
- For the SQA dataset, the model shows substantial improvements on all metrics, with an increase of at least 11 points in all metrics. The sequence accuracy improves from 28.1 to 40.4, and the average question accuracy increases from [More Information Needed] to [More Information Needed].
- The model achieves close to state-of-the-art performance for the WIKISQL dataset, with a denotation accuracy of 83.6.
- An ablation study on different embeddings shows that pre-training on tables, column and row embeddings are the most important factors in improving the model's quality.
- TAPAS achieves better or competitive results compared to state-of-the-art models in question answering and semantic parsing tasks across various datasets.

In summary, the google/tapas-base model performs well on different datasets, achieving high denotation accuracy, and shows improvements in various metrics. It outperforms or performs on par with other state-of-the-art models in question answering and semantic parsing tasks.

## Model Examination

The model google/tapas-base is a deep learning model based on BERT's encoder architecture. It has additional positional embeddings used to encode tabular structures. The table is flattened into a sequence of words, split into word pieces (tokens), and concatenated with the question tokens. The model also includes two classification layers for selecting table cells and aggregation operators.

The model has some limitations. It can parse compositional structures, but its expressivity is limited to aggregations over a subset of table cells. It cannot handle structures with multiple aggregations correctly. Additionally, the model can only handle single tables that can fit in memory. Very large tables or databases with multiple tables would pose a challenge. The table(s) could be compressed or filtered to include only relevant content, but this is left for future work.

In terms of explainability/interpretability, there is currently no specific information available in the references provided. [More Information Needed]

The code and pre-trained model for google/tapas-base are publicly available on GitHub at https://github.com/google-research/tapas. There are also released code and models to run TAPAS on TabFact for table entailment. A Colab notebook is provided for trying predictions on TabFact, and there is a new page describing the intermediate pre-training process.

The pre-training method for TAPAS extends BERT's masked language model objective to structured data. The model is pre-trained over millions of tables and related text segments crawled from Wikipedia. During pre-training, some tokens from the text segment and the table itself are masked, and the objective is to predict the original masked token based on the contextual information.

Overall, the model google/tapas-base provides a way to encode tabular structures using BERT's encoder architecture, but it has limitations in handling certain compositional structures and very large tables or databases. More work is needed in terms of explainability/interpretability.

## Environmental Impact

- **Hardware Type:** The model google/tapas-base is trained on a setup of 32 Cloud TPU v3 cores.
- **Software Type:** The model google/tapas-base is trained on structured data, specifically tables. It is a table parsing model that extends BERT's architecture to encode tables as input. The model is pre-trained on a large number of tables from Wikipedia to learn correlations between text and the table, as well as between the cells of a column and their header. The pre-training objective is a masked language model, where the model predicts the original masked token based on the textual and tabular context. The model also includes additional positional embeddings to encode the tabular structure. Inference is performed by flattening the table into a sequence of words, splitting words into word pieces (tokens), and concatenating the question tokens before the table tokens. The model has two classification layers for selecting table cells and aggregation operators. It can predict a minimal program by selecting a subset of table cells and a possible aggregation operation without generating logical forms. The model's performance on end tasks has been found to outperform or rival other semantic parsing datasets. However, there is no information provided about the specific software type used in training the model.
- **Hours used:** The amount of time used to train the model google/tapas-base is approximately 3 days for pre-training and around 10 hours for fine-tuning on the WIKISQL and WIKITQ datasets. For the SQA dataset, the fine-tuning process takes around 20 hours.
- **Cloud Provider:** The model google/tapas-base is trained on a setup of 32 Cloud TPU v3 cores.
- **Carbon Emitted:** To provide the amount of carbon emitted when training the model google/tapas-base, we need more information. The reference material does not mention the exact amount of carbon emitted during training.
## Technical Specification

### Model Architecture and Objective

The model architecture of google/tapas-base is based on BERT's encoder with additional positional embeddings used to encode tabular structure. The table is flattened into a sequence of words, and the question tokens are concatenated before the table tokens. The model includes two classification layers for selecting table cells and aggregation operators. Inference is performed by predicting a minimal program that selects a subset of table cells and a possible aggregation operation.

The objective of google/tapas-base is to reason over tables without generating logical forms. It predicts a minimal program by selecting a subset of table cells and a possible aggregation operation. This allows the model to learn operations from natural language without the need for a formalism. The model is pre-trained using BERT's masked language model objective, extended to structured data. During pre-training, the model masks tokens from both the text segment and the table itself and predicts the original masked tokens based on the contextual information.

[More Information Needed]

### Compute Infrastructure

The compute infrastructure for the model google/tapas-base is as follows:

1. Pre-training and fine-tuning were conducted on a setup of 32 Cloud TPU v3 cores.
2. The maximum sequence length used for both pre-training and fine-tuning was 512.
3. Pre-training took around 3 days, while fine-tuning took approximately 10 hours for WIKISQL and WIKITQ, and 20 hours for SQA.
4. The resource requirements for the model are similar to BERT-large.
5. The model was trained using a BERT-Large model as the starting point for pre-training.
6. The model was fine-tuned using hyper-parameters chosen through a black box Bayesian optimizer for WIKISQL and WIKITQ, and grid-search for SQA.
7. The standard BERT tokenizer was applied to questions, table cells, and headers using a vocabulary of 32k word pieces.
8. Numbers and dates were parsed in a similar way as in the Neural Programmer.
9. The evaluation scripts of WIKITQ and SQA were used to report denotation accuracy for these datasets.
10. For WIKISQL, the reference answer, aggregation operator, and cell coordinates were generated from the reference SQL using an SQL implementation running on JSON tables. However, the official WIKISQL evaluation script produced incorrect answers for approximately 2% of the examples.
11. The model has the same requirements as BERT, meaning that training the large model with a sequence length of 512 requires a TPU. Shorter sequences can be created using the option `max_seq_length` to make the model trainable on GPUs, but this may reduce accuracy. Another option is to reduce the batch size, but this may also affect accuracy.
12. Pre-trained checkpoints and the code for the model are publicly available at https://github.com/google-research/tapas.

Please note that the information provided above is based on the given references and may require further clarification or more specific details.

## Citation

```
@misc{jonathan-tapas,
    author = {Jonathan Herzig and
              Paweł Krzysztof Nowak and
              Thomas Müller and
              Francesco Piccinno and
              Julian Martin Eisenschlos},
    title  = {TAPAS: Weakly Supervised Table Parsing via Pre-training},
    url    = {https://arxiv.org/pdf/2004.02349.pdf}
}
```

