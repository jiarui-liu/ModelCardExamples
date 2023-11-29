# Model Card for roberta-base

RoBERTa-base is an improved version of BERT that achieves state-of-the-art performance on various natural language processing tasks by training the model longer, with bigger batches over more data, removing the next sentence prediction objective, training on longer sequences, and dynamically changing the masking pattern applied to the training data.

## Model Details

### Model Description

Model: roberta-base

Model Architecture: roberta-base uses the BERT-base architecture, which is based on the transformer architecture with L layers. Each block in roberta-base uses A self-attention heads and has a hidden dimension H.

Training Procedures: roberta-base is pretrained using the masked language modeling (MLM) objective and next sentence prediction. It is optimized with Adam using parameters β1 = 0.9, β2 = 0.999, ǫ = 1e-6, and L2 weight decay of 0.01. The learning rate is warmed up over the first 10,000 steps and then linearly decayed. roberta-base trains with a dropout of 0.1 on all layers and attention weights, and uses a GELU activation function. It is pretrained for S = 1,000,000 updates, with minibatches containing B = 256 sequences of maximum length T = 512.

Parameters: The number of parameters in roberta-base is 125M.

Important Disclaimers: [More Information Needed]

- **Developed by:** Yinhan Liu; Myle Ott; Naman Goyal; Jingfei Du; Mandar Joshi; Danqi Chen; Omer Levy; Mike Lewis; Luke Zettlemoyer; Veselin Stoyanov
- **Funded by:** The information provided does not mention the specific people or organizations that funded the project of the model roberta-base. [More Information Needed]
- **Shared by:** The contributors who made the model roberta-base available online as a GitHub repo are not mentioned in the provided references. [More Information Needed]
- **Model type:** The model roberta-base is a transformer-based model trained using masked language modeling (MLM) and next sentence prediction (NSP) objectives. It is a language model designed for natural language understanding tasks.
- **Language(s):** The model roberta-base uses and processes natural human language to perform tasks such as masked language modeling, sentence prediction, and disambiguation of pronouns. It achieves state-of-the-art results on various tasks and demonstrates improved performance through better training strategies.
- **License:** The license information for the model roberta-base is not provided in the given references. [More Information Needed]
- **Finetuned from model:** The model roberta-base is fine-tuned from the BERT-base model. The link to the BERT-base model is not provided in the given references.
### Model Sources

- **Repository:** https://github.com/jcpeterson/openwebtext
- **Paper:** https://arxiv.org/pdf/1907.11692.pdf
- **Demo:** The link to the demo of the model roberta-base is currently not provided in the given references. [More Information Needed]
## Uses

### Direct Use

The model `roberta-base` can be used without fine-tuning, post-processing, or plugging into a pipeline by following these steps:

1. Import the necessary libraries and load the model:
```python
import torch
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
roberta.eval()
```

2. Encode the input text using the `encode` method:
```python
tokens = roberta.encode('Input text')
```

3. Use the `predict` method to make predictions on a specific task:
```python
roberta.predict('task', tokens)
```

Here is an example of how to use the model for the MNLI task without any additional steps:
```python
tokens = roberta.encode('Roberta is a heavily optimized version of BERT.', 'Roberta is based on BERT.')
roberta.predict('mnli', tokens).argmax()  # Output: 2 (entailment)
```

Similarly, you can use the model for other tasks without the need for fine-tuning, post-processing, or plugging into a pipeline.

### Downstream Use

The roberta-base model can be used when fine-tuned for a specific task or when plugged into a larger ecosystem or app. Fine-tuning involves training the model on a task-specific dataset to adapt it for a particular downstream task.

To fine-tune the roberta-base model, you can use the Hugging Face Transformers library. Here is an example code snippet for fine-tuning the roberta-base model for a text classification task:

```
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW

# Load the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base')

# Prepare the training data
train_dataset = ...  # Your task-specific training dataset
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=16)

# Fine-tune the model
optimizer = AdamW(model.parameters(), lr=1e-5)
for epoch in range(3):
    for batch in train_dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        model.zero_grad()

# Save the fine-tuned model
model.save_pretrained('path/to/save/fine-tuned-model')
```

This code snippet demonstrates how to fine-tune the roberta-base model for a sequence classification task using the AdamW optimizer. You would need to replace `train_dataset` with your own task-specific training dataset.

Please note that this is just an example, and the exact fine-tuning process may vary depending on the specific task and dataset. It is recommended to refer to the official Hugging Face documentation and examples for more detailed instructions on fine-tuning the roberta-base model for different tasks.

[More Information Needed]

### Out-of-Scope Use

Model Card Description: roberta-base

Model Name: roberta-base

Description: roberta-base is a RoBERTa model that utilizes the BERT-base architecture. The model has been carefully developed and evaluated to improve performance on various natural language processing tasks such as GLUE, RACE, and SQuAD. It achieves state-of-the-art results without multi-task fine-tuning.

Model Architecture: BERT-base

Number of Parameters: 125 million

Download Link: [roberta.base.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz)

Misuse and Guidelines:

The roberta-base model should be used responsibly and ethically to ensure that it benefits society without causing harm. It is important to consider potential misuse scenarios and address them proactively. Here are some foreseeable misuses that users ought not to engage in with the roberta-base model:

1. Generating Inappropriate or Harmful Content: Users should not utilize the model to generate or spread hate speech, offensive content, or any form of harmful or misleading information. The model should be used responsibly to foster positive and inclusive communication.

2. Manipulating Information: Users should not use the model to manipulate or distort information for malicious purposes, such as spreading misinformation, propaganda, or engaging in political manipulation. The model should be used in a transparent and accountable manner.

3. Violating Privacy and Security: Users should not employ the model to infringe upon individuals' privacy or security. This includes unauthorized access to personal data, hacking, identity theft, or any other illegal activities.

4. Discrimination and Bias: Users should not discriminate against or marginalize individuals or groups based on their race, gender, religion, sexual orientation, or any other protected characteristics. The model should be used in a fair and unbiased manner, taking into account the potential biases present in the training data.

5. Legal and Ethical Compliance: Users should comply with all applicable laws and regulations while using the model. It is essential to respect intellectual property rights, privacy laws, and ethical guidelines.

It is crucial for users of the roberta-base model to be aware of the potential consequences of their actions and ensure that they align with ethical standards and societal norms. By using the model responsibly and considering the broader impact on society, we can maximize its benefits while minimizing any potential harm.

Please note that this model card description provides general guidelines, and users should exercise their judgment and consult with domain-specific experts to ensure responsible and ethical use of the roberta-base model.

[More Information Needed]

### Bias, Risks, and Limitations

The known or foreseeable issues stemming from the model roberta-base include:

1. Foreseeable Harms: There are no specific foreseeable harms mentioned in the references.

2. Misunderstandings: The model's performance may be misunderstood due to the complex training process and technical terminology used in the references.

3. Technical Limitations: The model's performance heavily relies on the amount of training data and the training procedure. The model may not generalize well to tasks outside of the pretraining objectives.

4. Sociotechnical Limitations: The model's impact on society may be limited by the availability and accessibility of large datasets required for training. The model's performance may also be affected by hyperparameter choices and training time.

In order to provide a more comprehensive analysis, further information is needed about the model's biases, fairness, privacy concerns, and potential ethical implications.

### Recommendations

Based on the references provided, the recommendations with respect to the foreseeable issues about the model roberta-base are as follows:

1. Use dynamic masking: Dynamic masking, where the masking pattern is generated every time a sequence is fed to the model, is recommended for pretraining with larger datasets or for more training steps. It has been found to be comparable or slightly better than static masking.

2. Consider the impact of hyperparameter choices: Hyperparameter choices have a significant impact on the final results of language model pretraining. It is important to carefully measure and consider the impact of different hyperparameters on the model's performance.

3. Explore and compare different model architectures: While RoBERTa uses the same architecture as BERT LARGE, it consistently outperforms both BERT LARGE and XLNet LARGE. This raises questions about the relative importance of model architecture and pretraining objective compared to other factors like dataset size and training time. Further exploration and comparison of different model architectures is recommended.

4. Evaluate combined impact of modifications: RoBERTa incorporates several modifications to the BERT pretraining procedure, including dynamic masking, FULL-SENTENCES without NSP loss, large mini-batches, and a larger byte-level BPE. It is recommended to evaluate the combined impact of these modifications on the model's performance.

In summary, the recommendations for roberta-base include using dynamic masking, carefully considering hyperparameter choices, exploring different model architectures, and evaluating the combined impact of modifications made to the pretraining procedure.

## Training Details

### Training Data

The training data for the model roberta-base consists of a combination of BOOKCORPUS and English WIKIPEDIA, along with three additional datasets: OPENWEBTEXT, STORIES, and CC-NEWS. The total uncompressed text used for pretraining is over 160GB. For more information, please refer to [tutorial for pretraining RoBERTa using your own data](README.pretraining.md).

### Training Procedure

#### Preprocessing

The preprocessing steps for the data of the model roberta-base involve tokenization and resizing/rewriting. Here are the details:

1. Tokenization: The data is tokenized using byte-level BPE (Byte Pair Encoding). This means that each word or subword is tokenized into a sequence of BPE tokens. The model outputs one feature vector per BPE token.

2. Resizing/Rewriting: The data is preprocessed by resizing or rewriting it, depending on the modality. Unfortunately, the exact details of this step are not provided in the references. [More Information Needed]

Overall, the data is preprocessed using tokenization with byte-level BPE, but further details about resizing or rewriting are required to provide a complete answer.

#### Training Hyperparameters

The training hyperparameters for the model roberta-base are as follows:

- Optimization algorithm: Adam
- Adam parameters: β1 = 0.9, β2 = 0.999, ǫ = 1e-6
- L2 weight decay: 0.01
- Learning rate warm-up: over the first 10,000 steps
- Peak learning rate: 1e-4
- Learning rate decay: linear decay after reaching the peak value
- Dropout: 0.1 on all layers and attention weights
- Activation function: GELU
- Pretraining updates: S = 1,000,000
- Minibatch size: B = 256 sequences
- Maximum sequence length: T = 512 tokens
- Training on full-length sequences
- Training with mixed-precision floating-point arithmetic
- RoBERTa pretrained for 100K steps over the BOOK-CORPUS plus WIKIPEDIA dataset
- RoBERTa pretrained over 160GB of text in total
- RoBERTa pretrained for significantly longer, with the number of pretraining steps increased from 100K to 300K

Note: For more detailed information on the training hyperparameters, please refer to the associated paper.

#### Speeds, Sizes, Times

The model card description for the model roberta-base is as follows:

The roberta-base model is a variant of the RoBERTa model, which is trained with a robust optimization approach called RoBERTa. It is trained with dynamic masking, full sentences without NSP loss, large mini-batches, and a larger byte-level BPE.

In terms of performance, the roberta-base model achieves state-of-the-art results on various benchmark datasets. On the SQuAD v1.1 development set, it matches the state-of-the-art set by XLNet. On the SQuAD v2.0 development set, it sets a new state-of-the-art, surpassing XLNet by 0.4 points (EM) and 0.6 points (F1).

The roberta-base model also performs well on other tasks such as RACE, HellaSwag, Commonsense QA, Winogrande, XNLI, and SuperGLUE. It achieves high accuracy and outperforms other models in these tasks.

Unfortunately, information about throughput, start or end time, and checkpoint sizes specific to the roberta-base model is not provided in the references. [More Information Needed]

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model roberta-base evaluates on the General Language Understanding Evaluation (GLUE) benchmark, which consists of 9 datasets for evaluating natural language understanding systems. These datasets include both single-sentence classification and sentence-pair classification tasks. The GLUE organizers provide training and development data splits, as well as a submission server and leaderboard that allows participants to evaluate and compare their systems on private held-out test data.

#### Factors

The model roberta-base is a state-of-the-art language model that has demonstrated superior performance on various tasks and datasets. It is based on the BERT architecture but incorporates several modifications and improvements to enhance its performance.

One of the key characteristics of roberta-base is its reliance on large amounts of text data for pretraining. It utilizes five English-language corpora, including BOOKCORPUS, English WIKIPEDIA, CC-NEWS, OPENWEBTEXT, and STORIES, totaling over 160GB of uncompressed text. This diverse and extensive training data allows the model to capture a wide range of language patterns and improve its end-task performance.

RoBERTa also implements several modifications to the BERT pretraining procedure, including dynamic masking, FULL-SENTENCES without NSP loss, large mini-batches, and a larger byte-level BPE. These changes, collectively known as the "RoBERTa for Robustly optimized BERT approach," contribute to the model's robustness and optimization, leading to improved performance compared to BERT LARGE and XLNet LARGE.

The model's performance has been evaluated on various datasets, including SQuAD v1.1 and SQuAD v2.0. On the SQuAD v1.1 development set, roberta-base matches the state-of-the-art set by XLNet. On the SQuAD v2.0 development set, it sets a new state-of-the-art, outperforming XLNet by 0.4 points (EM) and 0.6 points (F1).

While the model's performance has been impressive, it is important to consider the potential factors that may influence its behavior. These factors include the domain and context of the specific task, as well as population subgroups. Evaluating the performance of roberta-base across these factors can help uncover any disparities or biases in its performance. However, further information is needed to provide a more detailed analysis of such foreseeable characteristics and disparities in performance.

#### Metrics

The metrics used for evaluation in light of tradeoffs between different errors for the model roberta-base include:

1. For GLUE tasks: The model is evaluated based on various metrics like accuracy, F1 score, and exact match (EM) score on different GLUE tasks such as MNLI, QNLI, and SST-2.
2. For SQuAD 1.1 and 2.0: The model is evaluated based on EM and F1 scores on the SQuAD 1.1 and SQuAD 2.0 datasets.
3. For RACE: The model is evaluated based on accuracy on the RACE test set.
4. For HellaSwag: The model is evaluated based on overall accuracy, in-domain accuracy, zero-shot accuracy, ActivityNet accuracy, and WikiHow accuracy on the HellaSwag test set.
5. For public SQuAD 2.0 leaderboard: The model is evaluated based on its performance relative to other systems on the SQuAD 2.0 benchmark.

Please note that the specific tradeoffs between different errors are not mentioned in the given references.

### Results

The evaluation results of the model roberta-base are not explicitly mentioned in the given references. Therefore, for the evaluation results of roberta-base based on the Factors and Metrics, we need more information.

#### Summary

The evaluation results about the model roberta-base are as follows:

1. RoBERTa achieves state-of-the-art results on all 9 GLUE tasks development sets.
2. RoBERTa consistently outperforms both BERT LARGE and XLNet LARGE, which raises questions about the relative importance of model architecture and pretraining objective.
3. RoBERTa does not depend on multi-task finetuning, unlike most of the other top submissions on the GLUE leaderboard.
4. RoBERTa achieves state-of-the-art results on 4 out of 9 tasks and the highest average score on the GLUE leaderboard.
5. RoBERTa for Robustly optimized BERT approach is trained with dynamic masking, FULL-SENTENCES without NSP loss, large mini-batches, and a larger byte-level BPE.
6. RoBERTa outperforms all but one of the single model submissions on the SQuAD 2.0 leaderboard and is the top scoring system among those that do not rely on data augmentation.

Please note that there may be more detailed evaluation results available, but they are not provided in the given references.

## Model Examination

The model card description for the model roberta-base is as follows:

RoBERTa is a heavily optimized version of BERT, which is based on BERT and incorporates various modifications to improve performance. It has been evaluated on multiple benchmarks and has shown state-of-the-art performance on tasks such as SQuAD.

The RoBERTa model does not rely on any additional external training data, unlike other top-performing models like BERT and XLNet. Despite this, it outperforms most single model submissions on the SQuAD leaderboard and is the top scoring system among those that do not use data augmentation.

On the SQuAD v1.1 development set, RoBERTa matches the state-of-the-art set by XLNet. On the SQuAD v2.0 development set, RoBERTa sets a new state-of-the-art, surpassing XLNet's performance by 0.4 points (EM) and 0.6 points (F1).

RoBERTa uses the same masked language modeling pretraining objective and architecture as BERT LARGE. However, it consistently outperforms both BERT LARGE and XLNet LARGE, raising questions about the relative importance of model architecture and pretraining objective compared to other factors such as dataset size and training time.

Some improvements made to the BERT pretraining procedure are aggregated in RoBERTa, including dynamic masking, full-sentences without NSP loss, large mini-batches, and a larger byte-level BPE. These modifications contribute to the robust optimization of RoBERTa.

Unfortunately, the specific information about the experimental section related to explainability/interpretability for the model roberta-base is not provided in the given references. [More Information Needed]

## Environmental Impact

- **Hardware Type:** The hardware type on which the model roberta-base is trained is not mentioned in the provided references. Therefore, the answer is "[More Information Needed]".
- **Software Type:** The model roberta-base is trained using a software type called BERT.
- **Hours used:** The amount of time used to train the model `roberta-base` is not specified in the given references. Therefore, the answer is "[More Information Needed]".
- **Cloud Provider:** The model card for roberta-base is as follows:

Model Description:
RoBERTa-base is a variant of BERT that uses the BERT-base architecture. It is implemented in FAIRSEQ based on the original BERT optimization hyperparameters (Ott et al., 2019). The peak learning rate and number of warmup steps are tuned separately for each setting. Training is sensitive to the Adam epsilon term, and tuning it can improve performance and stability. Setting β2 = 0.98 has been found to improve stability with large batch sizes.

Experimental Setup:
RoBERTa-base is pretrained with sequences of at most 512 tokens. Unlike Devlin et al. (2019), short sequences are not randomly injected, and the model is trained only with full-length sequences. Training is done with mixed precision floating-point arithmetic on DGX-1 machines, each with 8 × 32GB Nvidia V100 GPUs interconnected by Infiniband (Micikevicius et al., 2018).

Performance:
RoBERTa-base outperforms all but one of the single model submissions on the SQuAD 2.0 leaderboard. It is the top scoring system among those that do not rely on data augmentation. RoBERTa-base consistently outperforms both BERT LARGE and XLNet LARGE, raising questions about the importance of model architecture and pretraining objective compared to dataset size and training time.

Model Details:
- Architecture: BERT-base
- Parameters: 125M
- Download: [roberta.base.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz)

Cloud Provider:
[More Information Needed]
- **Carbon Emitted:** The amount of carbon emitted when training the model roberta-base is not mentioned in the provided references. [More Information Needed].
## Technical Specification

### Model Architecture and Objective

The model architecture of roberta-base is based on the transformer architecture (Vaswani et al., 2017). It consists of L layers, where each layer has A self-attention heads and a hidden dimension of H.

The objective of roberta-base is the Masked Language Model (MLM). In this objective, a random sample of tokens in the input sequence is selected and replaced with the special token [MASK]. The model then predicts the masked tokens using a cross-entropy loss. 15% of the input tokens are uniformly selected for possible replacement, with 80% being replaced with [MASK], 10% left unchanged, and 10% replaced by a randomly selected vocabulary token.

Please note that the provided information is based on the given references, and for further details, additional information may be needed.

### Compute Infrastructure

The compute infrastructure information for the model roberta-base is not explicitly mentioned in the provided references. Therefore, we need more information to answer this question accurately.

## Citation

```
@misc{yinhan-roberta,
    author = {Yinhan Liu and
              Myle Ott and
              Naman Goyal and
              Jingfei Du and
              Mandar Joshi and
              Danqi Chen and
              Omer Levy and
              Mike Lewis and
              Luke Zettlemoyer and
              Veselin Stoyanov},
    title  = {RoBERTa: A Robustly Optimized BERT Pretraining Approach},
    url    = {https://arxiv.org/pdf/1907.11692.pdf}
}
```

