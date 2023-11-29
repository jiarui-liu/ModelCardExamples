# Model Card for xlm-roberta-base

The xlm-roberta-base model is a multilingual masked language model pre-trained on text in 100 languages, using a large Transformer model with 24 layers and 1024 hidden states, with a 250k vocabulary. It achieves state-of-the-art performance on cross-lingual classification, sequence labeling, and question answering tasks.

## Model Details

### Model Description

Model: xlm-roberta-base

Description:
xlm-roberta-base is a generic cross-lingual sentence encoder that obtains state-of-the-art results on many cross-lingual understanding (XLU) benchmarks. It is a transformer-based multilingual masked language model pre-trained on text in 100 languages. The model architecture is based on the BERT-base model with 12 layers and 768 hidden states. It has a vocabulary size of 250k tokens.

Training Objective:
The model is trained using the multilingual MLM (Masked Language Model) objective. It is trained to predict the masked tokens in the input text. The training data consists of monolingual data from each language. The text data is preprocessed using Sentence Piece with a unigram language model.

Training Data:
The model is trained on a large-scale dataset called CommonCrawl Corpus, which is built in 100 languages. The dataset is filtered using an internal language identification model in combination with fastText's language identification model. The dataset includes multiple CommonCrawl dumps for different languages, resulting in increased dataset sizes.

Training Procedure:
The model is trained for 1.5 Million updates on five-hundred 32GB Nvidia V100 GPUs with a batch size of 8192. The training utilizes SPM-preprocessed text data from Common-Crawl in 100 languages and samples languages with α = 0.3.

Parameters:
The xlm-roberta-base model has 270 million parameters.

Important Disclaimers:
[More Information Needed]

- **Developed by:** Alexis Conneau; Kartikay Khandelwal; Naman Goyal; Vishrav Chaudhary; Guillaume Wenzek; Francisco Guzmán; Edouard Grave; Myle Ott; Luke Zettlemoyer; Veselin Stoyanov; Facebook Ai
- **Funded by:** The people or organizations that fund the project of the model xlm-roberta-base are Facebook AI.
- **Shared by:** The contributors who made the model xlm-roberta-base available online as a GitHub repo are not mentioned in the provided references. [More Information Needed].
- **Model type:** The xlm-roberta-base model is a transformer-based multilingual masked language model that is trained using unsupervised multilingual pretraining with a large-scale dataset from CommonCrawl and achieves state-of-the-art performance on cross-lingual classification, sequence labeling, and question answering tasks.
- **Language(s):** The xlm-roberta-base model uses and processes multilingual natural human language.
- **License:** The license being used for the model xlm-roberta-base is not mentioned in the provided references. [More Information Needed]
- **Finetuned from model:** The base model for xlm-roberta-base is RoBERTa. Please refer to the repository for more information: [RoBERTa](https://github.com/pytorch/fairseq/tree/master/examples/roberta).
### Model Sources

- **Repository:** https://github.com/pytorch/fairseq/tree/master/examples/xlmr
- **Paper:** https://arxiv.org/pdf/1911.02116.pdf
- **Demo:** The link to the demo of the model xlm-roberta-base is not provided in the given references. [More Information Needed]
## Uses

### Direct Use

The model xlm-roberta-base can be used without fine-tuning, post-processing, or plugging into a pipeline by following these steps:

1. First, import the necessary libraries and load the model:
```python
import torch
xlmr = torch.hub.load('pytorch/fairseq:main', 'xlmr.large')
xlmr.eval()
```

2. Once the model is loaded, you can directly use it for various natural language processing tasks such as text classification, named entity recognition, sentiment analysis, etc., without the need for any additional fine-tuning.

3. To use the model, you can pass your input text directly to the model. The model will automatically tokenize the input text and generate the desired output. For example, to get the model's predictions for an input sentence, you can use the following code:
```python
input_text = "This is an example sentence."
output = xlmr.predict(input_text)
```
Note: The specific code for prediction may vary depending on the task you want to perform.

In summary, the xlm-roberta-base model can be readily used for various NLP tasks without the need for any fine-tuning, post-processing, or plugging into a pipeline. Just load the model, pass the input text, and retrieve the model's predictions.

### Downstream Use

The xlm-roberta-base model can be used for fine-tuning on specific tasks or integrated into larger ecosystems or applications. Fine-tuning involves training the model on a task-specific dataset to adapt it to perform well on that particular task. 

To fine-tune the xlm-roberta-base model, you can follow the code snippet below:

```python
from fairseq.models.roberta import XLMRModel

xlmr = XLMRModel.from_pretrained('/path/to/xlmr.large', checkpoint_file='model.pt')
xlmr.eval()  # disable dropout (or leave in train mode to finetune)
```

In this code snippet, we import the `XLMRModel` from the `fairseq.models.roberta` module. We then load the pre-trained xlm-roberta-base model using the `from_pretrained` method, specifying the path to the model and the checkpoint file. Finally, we disable dropout using the `eval()` method.

Once the model is fine-tuned, it can be used for various tasks such as cross-lingual classification, sequence labeling, and question answering. It can also be integrated into larger ecosystems or applications that require multilingual language understanding.

Please note that more specific information about how to fine-tune the xlm-roberta-base model for different tasks or integrate it into specific applications may be needed.

### Out-of-Scope Use

The xlm-roberta-base model is a multilingual masked language model trained on 2.5 TB of newly created clean CommonCrawl data in 100 languages. It has shown strong performance on cross-lingual classification, sequence labeling, and question answering tasks.

Considering the potential misuse of the model, there are a few aspects to address:

1. Bias and Discrimination: As with any language model, xlm-roberta-base can reflect biases present in the training data. It is crucial to mitigate biases related to gender, race, ethnicity, religion, and other protected characteristics. Care should be taken to ensure that the model does not amplify or perpetuate discriminatory or harmful stereotypes.

2. Misinformation and Propaganda: The model can generate text and responses, which may be exploited to spread misinformation or propaganda. It is essential to educate users about the responsible use of the model and discourage the generation of false or misleading information.

3. Privacy and Confidentiality: The model should not be used to process sensitive or confidential information without appropriate safeguards. User data and inputs should be handled securely, following best practices to protect privacy.

4. Legal and Ethical Compliance: Users should be reminded not to misuse the model for illegal activities, such as generating hate speech, harassment, or engaging in any form of harmful behavior. Clear guidelines and terms of use should be provided to ensure compliance with applicable laws and ethical standards.

5. Contextual Understanding: The xlm-roberta-base model may not fully grasp complex nuances and cultural contexts in different languages. Users should be aware of this limitation and avoid relying solely on the model's outputs without critical evaluation.

It is crucial to provide users with clear guidelines and instructions on the responsible use of the xlm-roberta-base model, encouraging ethical behavior, and discouraging any misuse that may have negative societal consequences. [More Information Needed]

### Bias, Risks, and Limitations

Based on the references provided, here are the known or foreseeable issues stemming from the xlm-roberta-base model:

1. Misunderstandings: Multilingual models like xlm-roberta-base may lead to misunderstandings due to the differences in languages and cultural nuances. The model may not accurately capture the context or subtle meanings of certain words or phrases in different languages.

2. Technical Limitations: The xlm-roberta-base model has limitations in handling low-resource languages. Although it shows improved performance for low-resource languages compared to previous models, there may still be challenges in accurately processing and understanding these languages due to limited training data.

3. Sociotechnical Limitations: The model has been trained on a large-scale dataset from CommonCrawl, which may introduce biases and reflect societal biases present in the data. These biases can potentially lead to unfair or discriminatory outcomes when the model is applied in real-world scenarios.

4. Foreseeable Harms: The xlm-roberta-base model may have the potential to reinforce or amplify existing biases present in the training data. This can result in biased predictions or discriminatory behavior when the model is used in applications such as text classification or natural language understanding.

It is important to carefully consider these issues and take appropriate steps to mitigate biases, address limitations, and ensure responsible and ethical usage of the xlm-roberta-base model in real-world applications. Further research and analysis are needed to fully understand the implications and potential harms associated with this model.

### Recommendations

Based on the references provided, there are several recommendations with respect to the foreseeable issues about the model xlm-roberta-base:

1. Improve the learning procedure: The references suggest that xlm-roberta-base, along with mBERT and XLM, are undertuned. Simple improvements in the learning procedure of unsupervised MLM can lead to much better performance.

2. Increase data for low-resource languages: Training on cleaned CommonCrawls can significantly increase the amount of data for low-resource languages. This has been shown to be effective for learning high-quality word embeddings in multiple languages.

3. Leverage parallel data: Translation language modeling (TLM) can be used to leverage parallel data and improve cross-lingual natural language inference (XNLI) performance. This approach has shown strong improvements in unsupervised machine translation and pretraining for sequence tasks.

4. Conduct comprehensive analysis: A comprehensive analysis of multilingual masked language models, particularly on tasks like XNLI and GLUE, can provide insights into the performance of xlm-roberta-base and its comparison with other models.

5. Address the trade-off between high and low resource languages: Consider the trade-off between positive transfer and capacity dilution when designing xlm-roberta-base. It is important to achieve good performance for both high and low resource languages at scale.

6. Improve performance on low-resource languages: xlm-roberta-base has the potential to perform particularly well on low-resource languages. Efforts should be made to further improve its performance on these languages, such as Swahili and Urdu.

7. Focus on unsupervised learning: Given the focus on unsupervised learning, it is recommended to use the unsupervised XLM-MLM objective rather than the supervised TLM objective for training xlm-roberta-base.

Overall, these recommendations aim to enhance the performance and applicability of xlm-roberta-base on multilingual tasks, improve its performance on low-resource languages, and address potential issues and trade-offs associated with multilingual modeling.

## Training Details

### Training Data

The training data for the xlm-roberta-base model consists of a clean CommonCrawl corpus in 100 languages, which is filtered using an internal language identification model in combination with fastText. The model is trained on 2.5T of this filtered CommonCrawl data. For more information, refer to Section 2 and Section 3 of the provided references.

### Training Procedure

#### Preprocessing

The data preprocessing for the xlm-roberta-base model involves tokenization and resizing/rewriting steps. Here are the details:

Tokenization:
The model uses Sentence Piece (SPM) for tokenization. SPM is applied directly on raw text data for all languages. This approach is used instead of language-specific preprocessing and byte-pair encoding. The use of SPM did not result in any loss in performance compared to other preprocessing methods.

Resizing/Rewriting:
There is no specific information provided about resizing or rewriting in the given references. Therefore, more information is needed to provide details about these steps.

In summary, the xlm-roberta-base model uses Sentence Piece for tokenization on raw text data. However, no information is available about resizing or rewriting in the provided references.

#### Training Hyperparameters

The training hyperparameters for the model xlm-roberta-base are not explicitly mentioned in the provided references. Therefore, I would need more information to provide the specific training hyperparameters for this model.

#### Speeds, Sizes, Times

The provided references do not directly mention the details about the throughput, start or end time, and checkpoint sizes of the xlm-roberta-base model. Hence, I need more information to provide a specific answer to this question.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model xlm-roberta-base evaluates on the following benchmarks and datasets:

1. Cross-lingual Natural Language Inference (XNLI): The XNLI dataset is used to evaluate cross-lingual transfer from English to other languages. It includes groundtruth dev and test sets in 15 languages, along with a ground-truth English training set.

2. Cross-lingual Question Answering: The MLQA benchmark is used for cross-lingual question answering. It extends the English SQuAD benchmark to include Spanish, German, Arabic, Hindi, Vietnamese, and Chinese languages.

3. Named Entity Recognition (NER): The model evaluates on the CoNLL-2002 and CoNLL-2003 datasets in English, Dutch, Spanish, and German for NER. It considers fine-tuning on the English set to evaluate cross-lingual transfer, fine-tuning on each set to evaluate per-language performance, or fine-tuning on all sets to evaluate multilingual learning.

4. GLUE Benchmark: The English performance of the model is evaluated on the GLUE benchmark. The GLUE benchmark consists of multiple classification tasks such as MNLI, SST-2, and QNLI.

Note: The information provided is based on the references provided and may not be exhaustive. Further information may be needed to provide a complete answer.

#### Factors

The foreseeable characteristics that will influence how the model xlm-roberta-base behaves are:

1. Domain and context: The model's performance is expected to vary depending on the specific domain and context it is applied to. The references do not provide specific details about the domains or contexts in which xlm-roberta-base was evaluated, so more information is needed to determine its behavior in different domains.

2. Population subgroups: The references do not provide explicit information about the evaluation of xlm-roberta-base on different population subgroups. Disaggregating the evaluation results across factors such as age, gender, ethnicity, or language proficiency could help uncover potential disparities in performance. However, without specific information, it is not possible to determine the model's behavior with respect to different population subgroups.

Overall, to fully understand how xlm-roberta-base behaves, further evaluation and analysis are needed, specifically focusing on domain-specific performance and potential disparities across population subgroups.

#### Metrics

The metrics used for evaluation of the model xlm-roberta-base include the following:

1. For crosslingual understanding:
   - Cross-lingual Natural Language Inference (NLI)
   - Named Entity Recognition (NER)
   - Question Answering

2. For English performance:
   - GLUE benchmark (comparing to other state-of-the-art models)
   - MNLI (Multi-Genre Natural Language Inference)
   - SST-2 (Stanford Sentiment Treebank)
   - QNLI (Question-answering NLI)

3. For multilingual performance:
   - CoNLL-2002 and CoNLL-2003 datasets in English, Dutch, Spanish, and German for NER
   - MLQA benchmark for cross-lingual Question Answering (English to Spanish, German, Arabic, Hindi, Vietnamese, and Chinese)

The metrics used for evaluation include F1 score, exact match (EM) score, and average accuracy. The performance of xlm-roberta-base is compared to baselines such as BERT Large and RoBERTa.

Please note that the exact values for the metrics are not provided in the references, so further information would be needed to know the precise performance of xlm-roberta-base on these metrics.

### Results

The xlm-roberta-base model has been evaluated on several factors and metrics. Here are the evaluation results:

1. Cross-lingual Natural Language Inference (XNLI):
   - The model has been evaluated on the XNLI dataset, which includes groundtruth dev and test sets in 15 languages.
   - The model has been trained on the English training set, which has been machine-translated to the remaining 14 languages to provide synthetic training data.
   - The model's performance has been evaluated on cross-lingual transfer from English to other languages.

2. Named Entity Recognition (NER):
   - The model has been evaluated on the CoNLL-2002 and CoNLL-2003 datasets in English, Dutch, Spanish, and German.
   - Three evaluation scenarios have been considered: cross-lingual transfer by fine-tuning on the English set, per-language performance by fine-tuning on each set, and multilingual learning by fine-tuning on all sets.
   - The performance has been measured using the F1 score and compared to baselines from Lample et al. (2016) and Akbik et al. (2018).

3. Cross-lingual Question Answering:
   - The model has been evaluated on the MLQA benchmark, which extends the English SQuAD benchmark to several languages including Spanish, German, Arabic, Hindi, Vietnamese, and Chinese.
   - The performance has been reported in terms of the F1 score and the exact match (EM) score for cross-lingual transfer from English.

4. GLUE Benchmark:
   - The model has been evaluated on the GLUE benchmark, which includes multiple classification tasks such as MNLI, SST-2, and QNLI.
   - The evaluation focuses on the English performance of the model.

Overall, the xlm-roberta-base model has shown strong performance on cross-lingual understanding tasks and natural language understanding tasks for multiple languages. It has achieved state-of-the-art results on various cross-lingual understanding benchmarks and has performed well on low-resource languages, improving accuracy in XNLI and NER tasks.

#### Summary

The evaluation results of the model xlm-roberta-base are as follows:

1. Cross-lingual Natural Language Inference (XNLI): The model is evaluated on cross-lingual transfer from English to other languages using the XNLI dataset. It achieves state-of-the-art results on this benchmark.

2. Named Entity Recognition (NER): The model is fine-tuned on datasets in English, Dutch, Spanish, and German to evaluate cross-lingual transfer and per-language performance. The F1 score is reported and compared to baselines, showing improved performance over previous XLM models for low-resource languages like Swahili and Urdu.

3. Cross-lingual Question Answering: The model is evaluated on the MLQA benchmark, which extends the English SQuAD benchmark to multiple languages. F1 score and exact match (EM) score are reported for cross-lingual transfer from English.

4. GLUE Benchmark: The model's English performance is evaluated using the GLUE benchmark, which includes multiple classification tasks. The model's performance is compared to other state-of-the-art models.

In summary, xlm-roberta-base performs well on cross-lingual understanding tasks, including natural language inference, named entity recognition, and question answering. It achieves state-of-the-art results on XNLI and shows improved performance for low-resource languages in NER. It also performs competitively on the GLUE benchmark.

## Model Examination

The xlm-roberta-base model is a transformer-based multilingual masked language model that has been pre-trained on text in 100 languages. It aims to improve cross-lingual language understanding (XLU) by training unsupervised crosslingual representations at a large scale.

According to references 1 and 3, the model has been evaluated on various tasks to assess its performance. It has been tested on the XNLI dataset, which is representative of its performance on other tasks. The model has also been evaluated on the GLUE benchmark to compare its performance against RoBERTa. The goal is to achieve strong performance on both cross-lingual understanding tasks and natural language understanding tasks for each language.

Reference 4 highlights that xlm-roberta-base is a generic cross-lingual sentence encoder that achieves state-of-the-art results on many cross-lingual understanding benchmarks. It has been trained on 2.5T of filtered CommonCrawl data in 100 languages.

Unfortunately, the provided references do not contain any information about the experimental section on explainability/interpretability for the xlm-roberta-base model. Therefore, more information is needed to answer this question.

## Environmental Impact

- **Hardware Type:** The hardware type that the model xlm-roberta-base is trained on is not explicitly mentioned in the provided references. Thus, the information about the hardware type is not available. [More Information Needed]
- **Software Type:** The model xlm-roberta-base is trained on a Transformer model with 12 layers, 768 hidden states, and a vocabulary size of 250K. It is trained using the multilingual MLM objective and trained for 1.5 Million updates on five-hundred 32GB Nvidia V100 GPUs. The model leverages the SPM-preprocessed text data from Common-Crawl in 100 languages and samples languages with α = 0.3. It is trained on 2.5T of filtered CommonCrawl data in 100 languages. It is a generic cross-lingual sentence encoder that obtains state-of-the-art results on many cross-lingual understanding (XLU) benchmarks. It does not use language embeddings and applies Sentence Piece tokenization directly on raw text data for all languages.
- **Hours used:** The amount of time used to train the model xlm-roberta-base is not provided in the given references. [More Information Needed]
- **Cloud Provider:** Based on the provided references, there is no specific information about the cloud provider used to train the model xlm-roberta-base. Therefore, the answer is "[More Information Needed]".
- **Carbon Emitted:** Based on the provided references, the amount of carbon emitted when training the model xlm-roberta-base is not mentioned. [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The xlm-roberta-base model is a transformer-based multilingual masked language model. It is trained on 2.5T of filtered CommonCrawl data in 100 languages. The training objective is to predict the masked tokens in the text using a multilingual MLM objective. The model architecture is similar to XLM-R Base, with 12 layers, a hidden size of 768, and attention heads of 12. The model has a vocabulary size of 250K and a total of 270M parameters. Additionally, the model does not use language embeddings, which allows it to better handle code-switching.

### Compute Infrastructure

The compute infrastructure for the xlm-roberta-base model is not explicitly mentioned in the provided references. Therefore, we need more information to determine the exact compute infrastructure used for training the model.

## Citation

```
@misc{alexis-unsupervised,
    author = {Alexis Conneau and
              Kartikay Khandelwal and
              Naman Goyal and
              Vishrav Chaudhary and
              Guillaume Wenzek and
              Francisco Guzmán and
              Edouard Grave and
              Myle Ott and
              Luke Zettlemoyer and
              Veselin Stoyanov and
              Facebook Ai},
    title  = {Unsupervised Cross-lingual Representation Learning at Scale},
    url    = {https://arxiv.org/pdf/1911.02116.pdf}
}
```

