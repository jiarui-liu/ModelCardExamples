# Model Card for google/tapas-tiny-finetuned-wtq

The model google/tapas-tiny-finetuned-wtq is a weakly supervised question answering model that reasons over tables without generating logical forms. It extends BERT's architecture with additional modifications and achieves better or competitive results in comparison to state-of-the-art models on semantic parsing datasets.

## Model Details

### Model Description

Model Card: google/tapas-tiny-finetuned-wtq

### Model Architecture
The google/tapas-tiny-finetuned-wtq model is based on BERT's encoder architecture. It incorporates additional positional embeddings to encode tabular structures. The table is flattened into a sequence of words, and the question tokens are concatenated with the table tokens. The model includes two classification layers for selecting table cells and aggregation operators to operate on the selected cells. The model also uses various positional embeddings, such as Previous Answer, Position ID, Segment ID, Column/Row ID, and Rank ID, to assist in processing the tabular data.

### Training Procedures
The model was pre-trained on a large number of tables from Wikipedia using a masked language model pre-training objective. The pre-training process allows the model to learn correlations between text and tables, as well as between the cells and their headers. The model was then fine-tuned on the Weakly Supervised Text-to-SQL (WTQ) dataset, using weak supervision and an end-to-end differentiable recipe.

### Parameters
[More Information Needed]

### Important Disclaimers
The model predicts a minimal program by selecting a subset of table cells and an aggregation operation without generating logical forms. The model can learn operations from natural language without the need for formal specifications. The model outperforms or rivals state-of-the-art models on semantic parsing datasets. However, it is important to note that the model's performance may vary depending on the specific use case and dataset.

Please note that this model card only provides a high-level overview of the model. For more detailed information, please refer to the original research papers cited in the references section.

- **Developed by:** Jonathan Herzig; Paweł Krzysztof Nowak; Thomas Müller; Francesco Piccinno; Julian Martin Eisenschlos
- **Funded by:** The people or organizations that funded the project of the model google/tapas-tiny-finetuned-wtq are:

1. Yasemin Altun
2. Srini Narayanan
3. Slav Petrov
4. William Cohen
5. Massimo Nicosia
6. Syrine Krichene
7. Jordan Boyd-Graber
8. Anonymous reviewers
9. Google PhD fellowship

Please note that the model card does not explicitly mention the funding sources for this specific model. However, it acknowledges the individuals and organizations for their feedback and support during the development process.
- **Shared by:** The contributors who made the model google/tapas-tiny-finetuned-wtq available online as a GitHub repo are [Jonathan Herzig](https://github.com/jonathanherzig), [Paweł Krzysztof Nowak](https://github.com/paweljn), [Thomas Müller](https://github.com/tmueller91), [Francesco Piccinno](https://github.com/frapicc00), and [Julian Martin Eisenschlos](https://github.com/julian-e).
- **Model type:** The model google/tapas-tiny-finetuned-wtq is a pre-trained model that uses BERT's encoder architecture and additional positional embeddings to encode tabular structure, and it is fine-tuned on question answering tasks over tables using weak supervision. It falls under the training method of pre-training and fine-tuning, the machine learning type of deep learning, and the modality of text and tabular data.
- **Language(s):** The model google/tapas-tiny-finetuned-wtq uses natural human language to answer questions based on tabular data without the need for explicit logical forms or formalism.
- **License:** The license information for the model google/tapas-tiny-finetuned-wtq is not provided in the given references. [More Information Needed]
- **Finetuned from model:** The model google/tapas-tiny-finetuned-wtq is fine-tuned from another base model, but the name and link to that base model are not provided in the given references. [More Information Needed]
### Model Sources

- **Repository:** https://github.com/google-research/tapas
- **Paper:** https://arxiv.org/pdf/2004.02349.pdf
- **Demo:** The demo of the model google/tapas-tiny-finetuned-wtq can be found at this link: [custom table question answering widget](https://huggingface.co/google/tapas-tiny-finetuned-wtq).
## Uses

### Direct Use

The model "google/tapas-tiny-finetuned-wtq" can be used without fine-tuning, post-processing, or plugging into a pipeline. It is a pre-trained model that can directly be used for question answering over tables.

To use the model, you can follow this code snippet:

```python
from transformers import TapasTokenizer, TapasForQuestionAnswering

tokenizer = TapasTokenizer.from_pretrained("google/tapas-tiny-finetuned-wtq")
model = TapasForQuestionAnswering.from_pretrained("google/tapas-tiny-finetuned-wtq")

# Prepare the input
table = [["Country", "Capital"], ["France", "Paris"], ["Germany", "Berlin"]]
queries = ["What is the capital of France?", "Which country has Berlin as its capital?"]

inputs = tokenizer(table=table, queries=queries, padding="max_length", truncation=True, return_tensors="pt")

# Perform inference
outputs = model(**inputs)

# Extract the answers
answer_coordinates = outputs.answer_coordinates.tolist()
answer_texts = [table[row][column] for row, column in answer_coordinates]

print(answer_texts)
```

Please note that the input should consist of a table in the form of a list of lists, where each inner list represents a row in the table. The queries should be provided as a list of strings. The code snippet uses the TapasTokenizer and TapasForQuestionAnswering classes from the Hugging Face Transformers library.

This code will tokenize the input, pass it through the pre-trained model, and extract the answer coordinates. The answer texts are then extracted from the table based on the answer coordinates.

Please note that this code assumes that you have the necessary dependencies installed, including the Transformers library and its dependencies.

### Downstream Use

The model google/tapas-tiny-finetuned-wtq is a fine-tuned version of TAPAS specifically for the WTQ (WikiTableQuestions) dataset. TAPAS (Table Parser) is a weakly supervised question answering model that reasons over tables without generating logical forms. It predicts a minimal program by selecting a subset of table cells and an aggregation operation to be executed on them.

When fine-tuned for a task, the google/tapas-tiny-finetuned-wtq model can be used to answer questions based on tabular data. It takes as input a table and a natural language question and outputs the answer. The model can be plugged into a larger ecosystem or app that requires question answering capabilities over tabular data.

To use the model, you need to provide the table and question as input. Here's an example code snippet for using the model:

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_name = "google/tapas-tiny-finetuned-wtq"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

table = "Your table data"  # Replace with your table data
question = "Your question"  # Replace with your question

# Tokenize inputs
inputs = tokenizer(table, question, return_tensors="pt")

# Get model prediction
outputs = model(**inputs)
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(outputs["output"]))

print("Answer:", answer)
```

Please note that you need to replace "Your table data" and "Your question" with the actual table data and question you want to use. Additionally, make sure you have the required dependencies installed and the model is downloaded or loaded.

For more advanced usage and fine-tuning the model on different datasets or tasks, please refer to the model's GitHub repository (https://github.com/google-research/tapas) and the associated task data mentioned in the references.

[More Information Needed]

### Out-of-Scope Use

The model google/tapas-tiny-finetuned-wtq has the potential to be misused in certain ways. One foreseeable misuse of the model is using it to automate the extraction of sensitive or private information from tables without proper authorization or consent. This could include extracting personal data, financial information, or any other confidential data that should be protected.

It is important to note that the model itself does not have the ability to distinguish between public and private data, and it is the responsibility of the users to ensure that they are using the model in an ethical and legal manner. Users of the model should not engage in any activities that violate privacy laws or regulations.

Additionally, the model may also be misused to generate misleading or false information by manipulating the queries or the table data. This could lead to the spread of misinformation or the creation of fake news. Users should exercise caution and responsibility when using the model to ensure that the information generated is accurate and reliable.

To mitigate these potential misuses, it is crucial to provide clear guidelines and ethical considerations to users of the model. This includes promoting responsible use, respecting privacy rights, and adhering to legal and ethical standards. It is also important to educate users about the limitations of the model and the potential risks associated with its misuse.

In conclusion, users of the model google/tapas-tiny-finetuned-wtq should not misuse it to extract sensitive or private information without proper authorization and consent, and they should also avoid using it to generate misleading or false information. Responsible and ethical use of the model is essential to ensure that it benefits society without causing harm.

### Bias, Risks, and Limitations

The known or foreseeable issues stemming from the model google/tapas-tiny-finetuned-wtq are as follows:

1. In some tables, the WIKISQL dataset contains "REAL" numbers stored in "TEXT" format. This can lead to incorrect results for comparison and aggregation examples. This issue penalizes systems that do their own execution instead of producing an SQL query, as the numbers in the table are not represented correctly. This can affect the accuracy of the model's execution of queries.

2. The model is trained on large-scale data of text-table pairs and fine-tuned on semantic parsing datasets using weak supervision. While this approach yields competitive results, it may still have limitations in terms of accuracy and generalizability.

3. The model is currently designed to handle single tables as context. Future work aims to extend the model to represent a database with multiple tables effectively and handle large tables. This limitation may affect the model's performance when dealing with complex scenarios involving multiple tables.

4. The model's performance in predicting numbers is evaluated using a soft metric that relaxes the accuracy measure. While this metric allows for some flexibility and accounts for approximate guesses, it may not capture the true accuracy of the model's number predictions.

5. The model has certain requirements, such as the need for TPUs when training the large model with a sequence length of 512. There are options to create shorter sequences or reduce batch size, but these adjustments may impact the model's accuracy.

6. There is a small change in the WTQ training example creation, where questions with ambiguous cell matches are discarded. This change improves denotation accuracy but may affect the overall performance on certain types of questions.

7. The model includes an option to filter table columns by textual overlap with the question using the HEM method. However, the limitations and effects of this filtering method are not explicitly mentioned in the provided references.

It's important to note that this answer is based on the given references and may not cover all possible issues. Further analysis and evaluation may be required to uncover additional limitations or potential harms.

### Recommendations

Based on the references provided, the following recommendations can be made with respect to the foreseeable issues about the model google/tapas-tiny-finetuned-wtq:

1. **Training Objective**: The model is pre-trained on large-scale data of text-table pairs using language modeling and masked language modeling objectives. It is recommended to ensure that the pre-training objectives align with the desired task and domain to avoid any bias or performance limitations. Further analysis is needed to assess the suitability of these objectives for the specific use case.

2. **Weak Supervision**: TAPAS model can be fine-tuned on semantic parsing datasets using weak supervision. It is essential to carefully design and curate the weak supervision data to ensure accurate and reliable training. Additional research is required to explore the potential limitations and biases introduced by weak supervision in the fine-tuning process.

3. **Logical Form Generation**: TAPAS model does not explicitly generate logical forms, which may limit its ability to handle complex queries or tasks that require explicit symbolic operations. It is important to evaluate and understand the impact of this limitation on the model's performance for specific use cases.

4. **Coverage and Aggregation**: TAPAS model supports aggregation operators over selected cells, providing more coverage compared to previous models. However, it is crucial to assess the model's accuracy and robustness when predicting aggregations over table cells, as this functionality may introduce potential errors or biases in the results.

5. **Transfer Learning**: TAPAS demonstrates promising results in transfer learning, achieving higher accuracy compared to state-of-the-art models when pre-trained on different datasets. It is advisable to investigate the suitability and potential benefits of transfer learning for specific tasks and domains to leverage the performance gains achieved by TAPAS.

6. **Data Quality**: Recent updates to the WTQ training example creation process have improved denotation accuracy by discarding questions with ambiguous cell matches. It is recommended to ensure high-quality training data, including accurate annotations and proper handling of ambiguous cases, to enhance the model's performance and reliability.

7. **Filtering Table Columns**: The addition of an option to filter table columns based on textual overlap with the question can improve the model's performance. It is suggested to explore the effectiveness and impact of this feature for specific use cases and assess its potential benefits in terms of accuracy and relevance.

Please note that further investigation and analysis are required to fully understand and address the foreseeable issues and recommendations related to the model google/tapas-tiny-finetuned-wtq. [More Information Needed].

## Training Details

### Training Data

The training data for the model google/tapas-tiny-finetuned-wtq consists of complex questions on Wikipedia tables from the WIKITQ dataset, where the questions were composed by crowd workers and verified by other crowd workers. The model is trained to predict cell selection or scalar answers based on the denotations provided in the dataset. For more information, please refer to References 2 and 3.

### Training Procedure

#### Preprocessing

The preprocessing for the data of the model google/tapas-tiny-finetuned-wtq involves tokenization, resizing, and rewriting. Here are the details:

1. Tokenization: The standard BERT tokenizer is applied to questions, table cells, and headers using a vocabulary of 32k word pieces. Numbers and dates are parsed in a similar way as in the Neural Programmer (Neelakantan et al., 2017).

2. Resizing/Rewriting: To ensure efficient pre-training, the word piece sequence length is restricted to a certain budget (e.g., 128 in the final experiments). This means that the combined length of tokenized text and table cells must fit into this budget. To achieve this, a snippet of 8 to 16 word pieces is randomly selected from the associated text. For the table, the first word of each column name and cell is added, and then additional words are added turn-wise until the word piece budget is reached.

In summary, the data preprocessing for the model google/tapas-tiny-finetuned-wtq involves tokenizing the input using the BERT tokenizer and resizing/rewriting the input to fit within a certain word piece budget.

Please note that further details about the implementation or code examples are not provided in the given references.

#### Training Hyperparameters

The reference does not provide specific training hyperparameters for the model google/tapas-tiny-finetuned-wtq. Therefore, I cannot provide the detail training hyperparameters for this model. [More Information Needed]

#### Speeds, Sizes, Times

The model google/tapas-tiny-finetuned-wtq, which is based on TAPAS, has been fine-tuned on the WTQ 1.0 dataset. 

Unfortunately, there is no specific information provided about the throughput, start or end time, or checkpoint sizes of this model in the given references. Hence, we would need more information to provide those details.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model google/tapas-tiny-finetuned-wtq evaluates on the WIKITQ dataset. WIKITQ consists of complex questions on Wikipedia tables, where crowd workers were asked to compose a series of complex questions that include comparisons, superlatives, aggregation, or arithmetic operations. The questions were then verified by other crowd workers. The official evaluation script of WIKITQ is used to report the denotation accuracy for this dataset.

#### Factors

The foreseeable characteristics that will influence how the model google/tapas-tiny-finetuned-wtq behaves include the domain and context of the data, as well as the population subgroups. 

Regarding the domain and context, the model has been trained on structured data from tables and related text segments crawled from Wikipedia. Therefore, its performance is expected to be influenced by the specific types of tables and text it has been exposed to during pre-training. If the input data deviates significantly from the domains and contexts seen during pre-training, the model's performance may be affected.

In terms of population subgroups, it is important to evaluate the model's performance across different factors to uncover any disparities. Disaggregating the evaluation based on factors such as age, gender, race, or language can help identify if the model's performance varies across different subgroups. This evaluation is crucial to ensure fairness and mitigate any potential biases in the model's predictions.

To evaluate the model's performance across factors, it is recommended to analyze the model's accuracy, as well as metrics such as sequence accuracy and average question accuracy. By examining these metrics across different population subgroups, any disparities in the model's performance can be identified and addressed.

Unfortunately, the provided references do not explicitly mention the evaluation disaggregated across factors or specific disparities in performance. Therefore, more information is needed to provide a comprehensive answer to this question.

#### Metrics

The metrics used for evaluation in light of tradeoffs between different errors for the model google/tapas-tiny-finetuned-wtq are not explicitly mentioned in the provided references. [More Information Needed]

### Results

Based on the provided references, the evaluation results for the model google/tapas-tiny-finetuned-wtq are as follows:

1. Factors: 
   - Dataset: WIKISQL, WIKITQ, SQA
   - Training Setup: Pre-training and fine-tuning on a setup of 32 Cloud TPU v3 cores with a maximum sequence length of 512
   - Resource Requirements: Similar to BERT-large

2. Metrics:
   - WIKISQL: Close to state-of-the-art performance with a denotation accuracy of 83.6 (vs. 83.9 in Min et al., 2019)
   - WIKITQ: Model trained only from the original training data achieves 42.6, surpassing similar approaches. Pre-training on WIKISQL leads to an accuracy of 48.7.
   - SQA: TAPAS leads to substantial improvements on all metrics, improving all metrics by at least 11 points. Sequence accuracy increased from 28.1 to 40.4, and average question accuracy improved.

3. Additional Information:
   - Supervision: Weakly supervised setting with the option for full supervision by providing gold aggregation operators and selected cell annotations.
   - Training Time: Pre-training takes around 3 days, fine-tuning takes around 10 hours for WIKISQL and WIKITQ, and 20 hours for SQA.
   - Model Performance: Performs better or on par with other semantic parsing and question answering models on different datasets.
   - Transfer Learning: Transfer learning from WIKISQL to WIKITQ achieves an accuracy of 48.7, 4.2 points higher than the state-of-the-art.
   - Number Accuracy: The model shows good performance at guessing numbers close to the target, with an overall accuracy of 74.5 instead of 71.4 and an accuracy of 80.5 instead of 53.9 for numbers.

Please note that the specific evaluation results for google/tapas-tiny-finetuned-wtq are not provided in the references.

#### Summary

The evaluation results for the model google/tapas-tiny-finetuned-wtq are as follows:

1. On the conversational SQA dataset, TAPAS improves the state-of-the-art accuracy from 55.1 to 67.2.
2. TAPAS achieves on par performance on the WIKISQL and WIKITQ datasets.
3. Transfer learning from WIKISQL to WIKITQ achieves an accuracy of 48.7, which is 4.2 points higher than the state-of-the-art.
4. The model trained only from the original training data reaches an accuracy of 42.6 on the WIKITQ dataset.
5. Pre-training the model on WIKISQL or SQA leads to improved accuracies of 48.7 and 48.8, respectively.
6. TAPAS improves all metrics on the SQA dataset by at least 11 points, with sequence accuracy increasing from 28.1 to 40.4 and average question accuracy improving.

Overall, TAPAS achieves better or competitive results compared to other semantic parsing and question answering models on different datasets. [More Information Needed]

## Model Examination

The model google/tapas-tiny-finetuned-wtq, also known as TAPAS (Table Parser), is a weakly supervised question answering model designed to reason over tables without generating logical forms. It predicts a minimal program by selecting a subset of table cells and an aggregation operation to be executed on top of them. This model extends BERT's architecture with additional components.

The TAPAS model is based on previous works that have proposed end-to-end differentiable models for training from weak supervision. However, TAPAS is unique in its ability to learn operations from natural language without the need to specify them in some formalism.

The expressivity of TAPAS is limited to a form of aggregation over a subset of table cells, and it cannot handle structures with multiple aggregations. Despite this limitation, TAPAS has been successful in parsing three different datasets.

The TAPAS model follows a similar line of work as Cho et al. (2018), which proposes a supervised model that predicts the relevant rows, columns, and aggregation operations sequentially. However, TAPAS has a simpler architecture and provides more coverage by supporting aggregation operators over selected cells.

In terms of pre-training, TAPAS extends masked language modeling for table representations by masking table cells or text segments. It also utilizes intermediate pre-training, which is described in detail in the provided references.

Unfortunately, there is no specific information available regarding the work on explainability/interpretability for the google/tapas-tiny-finetuned-wtq model. [More Information Needed].

## Environmental Impact

- **Hardware Type:** The model google/tapas-tiny-finetuned-wtq is trained on a setup of 32 Cloud TPU v3 cores.
- **Software Type:** The model google/tapas-tiny-finetuned-wtq is trained on TAPAS, a weakly supervised question answering model that reasons over tables without generating logical forms. TAPAS extends BERT's architecture with additional modifications and positional embeddings to encode the tabular structure. It predicts a minimal program by selecting a subset of the table cells and a possible aggregation operation to be executed on top of them. This allows TAPAS to learn operations from natural language without the need for a formalism. However, the specific software type that the model is trained on is not mentioned in the references provided. [More Information Needed]
- **Hours used:** The amount of time used to train the model google/tapas-tiny-finetuned-wtq is not provided in the given references. [More Information Needed]
- **Cloud Provider:** The model google/tapas-tiny-finetuned-wtq is trained on a setup of 32 Cloud TPU v3 cores.
- **Carbon Emitted:** The information about the amount of carbon emitted when training the model google/tapas-tiny-finetuned-wtq is not provided in the given references. [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The model google/tapas-tiny-finetuned-wtq is a weakly supervised question answering model that reasons over tables without generating logical forms. It extends BERT's architecture with additional positional embeddings to encode tabular structure.

The model's architecture is based on BERT's encoder, with the table flattened into a sequence of words. The question tokens are concatenated with the table tokens, and a separator token is added between them. The model includes two classification layers for selecting table cells and aggregation operators.

The model uses different kinds of positional embeddings, including position ID (the index of the token in the flattened sequence), segment ID (0 for the question and 1 for the table header and cells), column/row ID (the index of the column/row that the token appears in), and rank ID (an embedding based on the numeric rank of column values).

The objective of the model is to predict a minimal program by selecting a subset of the table cells and a possible aggregation operation to be executed on top of them. This allows the model to learn operations from natural language without the need for a formal specification.

The model has been evaluated on three different semantic parsing datasets and has shown better or competitive results compared to other semantic parsing and question answering models. Transfer learning from WIKISQL to WIKITQ achieves higher accuracy than the state-of-the-art.

Code and checkpoints for training the model can be found in the paper "TAPAS: Weakly Supervised Table Parsing via Pre-training" (how-to-cite-tapas).

### Compute Infrastructure

The compute infrastructure for the model google/tapas-tiny-finetuned-wtq is as follows:

1. Pre-training and fine-tuning were performed on a setup of 32 Cloud TPU v3 cores.
2. The maximum sequence length for both pre-training and fine-tuning was set to 512.
3. Pre-training took approximately 3 days, while fine-tuning took around 10 hours for WIKISQL and WIKITQ, and 20 hours for SQA.
4. The resource requirements of the model are similar to BERT-large.
5. Hyper-parameters for fine-tuning were selected using a black box Bayesian optimizer for WIKISQL and WIKITQ, and grid-search for SQA.
6. Pre-training was started from a pre-trained standard text BERT model, with additional embeddings randomly initialized, to enhance convergence on the held-out set.

Please note that more information is needed about the exact specifications of the Cloud TPU v3 cores used and the batch sizes employed during training.

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

