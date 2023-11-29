# Model Card for facebook/xlm-v-base

The model facebook/xlm-v-base is a multilingual language model with a one million token vocabulary trained on 2.5TB of data from Common Crawl. It outperforms the XLM-R model on various tasks and provides semantically meaningful and shorter tokenizations.

## Model Details

### Model Description

Model Name: facebook/xlm-v-base

Description:
The facebook/xlm-v-base model is a multilingual language model with a one million token vocabulary. It was introduced in the paper "XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models" by Davis Liang, Hila Gonen, Yuning Mao, Rui Hou, Naman Goyal, Marjan Ghazvininejad, Luke Zettlemoyer, and Madian Khabsa. This model aims to address the vocabulary bottleneck in multilingual models like XLM-R by reducing over-tokenization for low-resource languages and assigning vocabulary capacity to achieve sufficient coverage for each individual language. 

Architecture:
The architecture of facebook/xlm-v-base is not specified in the provided references. [More Information Needed]

Training Procedures:
The model is pretrained on the CC100 dataset using the same training procedure as XLM-R. It uses the Masked Language Model (MLM) task with a standard masking rate of 15%. The training process includes using the Adam optimizer with default parameters and a learning rate of 6e-4. The model is trained for a total of 1.5 million iterations, with each batch consisting of examples concatenated up to a maximum sequence length of 512. [More Information Needed]

Parameters:
The specific parameters of the facebook/xlm-v-base model are not provided in the references. [More Information Needed]

Important Disclaimers:
- The tokenizer in 🤗 Transformers should output the same ids/subtokens as the `fairseq` tokenizer, as mentioned in the references.
- The references highlight the improvements of XLM-V over XLM-R on various tasks such as natural language inference (XNLI), question answering (MLQA, XQuAD, TyDiQA), and named entity recognition (WikiAnn). However, specific evaluation results and comparisons are not provided. [More Information Needed]

Please note that the provided information may be incomplete. For additional details, please refer to the original paper or further documentation.

- **Developed by:** Davis Liang; Hila Gonen; Yuning Mao; Rui Hou; Naman Goyal; Marjan Ghazvininejad; Luke Zettlemoyer; Madian Khabsa; Meta Ai
- **Funded by:** The people or organizations that fund the project of the model `facebook/xlm-v-base` are:

- Davis Liang
- Hila Gonen
- Yuning Mao
- Rui Hou
- Naman Goyal
- Marjan Ghazvininejad
- Luke Zettlemoyer
- Madian Khabsa
- Meta AI
- **Shared by:** The contributors who made the model `facebook/xlm-v-base` available online as a GitHub repo are:

1. Davis Liang
2. Hila Gonen
3. Yuning Mao
4. Rui Hou
5. Naman Goyal
6. Marjan Ghazvininejad
7. Luke Zettlemoyer
8. Madian Khabsa
9. Meta AI

Please note that the above information is based on the available references.
- **Model type:** The model facebook/xlm-v-base is a multilingual language model pretrained using the Masked Language Model (MLM) task, trained on the CC100 dataset with a sampling temperature of 0.3, and fine-tuned with various downstream tasks. It is trained using the XLM-R training procedure, and it overcomes the vocabulary bottleneck in multilingual masked language models. [More Information Needed]
- **Language(s):** The model facebook/xlm-v-base processes natural human language from a wide range of languages, as it is a multilingual language model trained on 176 languages from the WikiANN dataset.
- **License:** The license being used for the model `facebook/xlm-v-base` is not mentioned in the provided references. [More Information Needed]
- **Finetuned from model:** The model facebook/xlm-v-base is a multilingual language model with a one million token vocabulary. It was introduced in the paper "XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models" by Davis Liang, Hila Gonen, Yuning Mao, Rui Hou, Naman Goyal, Marjan Ghazvininejad, Luke Zettlemoyer, and Madian Khabsa.

According to the abstract of the paper, XLM-V overcomes the vocabulary bottleneck of multilingual models like XLM-R by de-emphasizing token sharing between languages with little lexical overlap and assigning vocabulary capacity to achieve sufficient coverage for each individual language. Tokenizations using XLM-V's vocabulary are typically more semantically meaningful and shorter compared to XLM-R. XLM-V outperforms XLM-R on every task tested, including natural language inference, question answering, and named entity recognition.

The model was trained on 2.5TB of data from Common Crawl, similar to XLM-R. It is particularly effective on low-resource language tasks and outperforms XLM-R by 11.2% and 5.8% absolute on MasakhaNER and Americas NLI, respectively.

Regarding whether the model is fine-tuned from another model, there is no information available in the provided references. Therefore, more information is needed to answer this question.
### Model Sources

- **Repository:** https://github.com/stefan-it/xlm-v-experiments
- **Paper:** https://arxiv.org/pdf/2301.10472.pdf
- **Demo:** The model card description for the model `facebook/xlm-v-base` is as follows:

### Model Description

XLM-V is a multilingual language model with a one million token vocabulary trained on 2.5TB of data from Common Crawl. It was introduced in the paper titled "XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models" by Davis Liang, Hila Gonen, Yuning Mao, Rui Hou, Naman Goyal, Marjan Ghazvininejad, Luke Zettlemoyer, and Madian Khabsa. This model aims to address the over-tokenization issue in low-resource languages and provides semantically meaningful tokenizations with reduced average sequence length compared to XLM-R. XLM-V outperforms XLM-R on various tasks such as natural language inference (XNLI), question answering (MLQA, XQuAD, TyDiQA), and named entity recognition (WikiAnn).

### Model Card Updates

I will serve as the contact person for any updates to the model card.

### Demo

To access the demo of the model `facebook/xlm-v-base`, please visit the following link: [Demo for facebook/xlm-v-base](https://huggingface.co/facebook/xlm-v-base)
## Uses

### Direct Use

The model "facebook/xlm-v-base" can be used without fine-tuning, post-processing, or plugging into a pipeline. It can directly accept input text and generate predictions or embeddings. Here's an example code snippet to demonstrate its usage:

```python
from transformers import XLMTokenizer, XLMModel

# Instantiate the tokenizer and model
tokenizer = XLMTokenizer.from_pretrained("facebook/xlm-v-base")
model = XLMModel.from_pretrained("facebook/xlm-v-base")

# Input text
input_text = "Hello, how are you?"

# Tokenize the input text
tokens = tokenizer(input_text, return_tensors="pt")

# Generate embeddings
outputs = model(**tokens)

# Access the last hidden state
embeddings = outputs.last_hidden_state

# Print the embeddings
print(embeddings)
```

In this example, we first instantiate the tokenizer and model using the "facebook/xlm-v-base" pre-trained weights. Then, we provide an input text and tokenize it using the tokenizer. Next, we pass the tokenized inputs to the model and obtain the output embeddings. Finally, we can access and manipulate the embeddings as needed.

Please note that this code snippet assumes you have already installed the "transformers" library and its dependencies. If not, you can install it using `pip install transformers`.

Please let me know if you need further assistance or information.

### Downstream Use

The model `facebook/xlm-v-base` can be used when fine-tuned for a specific task or when integrated into a larger ecosystem or application. When fine-tuning the model, it is recommended to follow the steps mentioned in the references.

To fine-tune the model for a specific task, you can use the `flair-fine-tuner.py` script provided in the repository. This script fine-tunes the model on the English WikiANN dataset for Named Entity Recognition (NER). The script expects a model configuration file as the first input argument, and you can find the configuration files under the `./configs` folder. The fine-tuning process uses a sequence length of 512.

Here is an example code snippet to start fine-tuning XLM-V for NER using the `flair-fine-tuner.py` script:
```
python flair-fine-tuner.py <model_configuration_file>
```

Please note that this code snippet assumes that you have the necessary dependencies and dataset in place. For more details on the fine-tuning process and hyperparameters, please refer to the referenced papers and scripts.

[More Information Needed]

### Out-of-Scope Use

The model `facebook/xlm-v-base` is a powerful multilingual model that has been uploaded to the 🤗 Transformers Model Hub and is available under the Meta AI organization. It has also been added to the 🤗 Transformers Documentation.

While the strengths of XLM-V are clear, there are potential misuses of the model that should be addressed to ensure responsible use. One potential misuse is using the model for malicious purposes, such as generating harmful or misleading content. This could lead to the spread of misinformation or the creation of harmful narratives.

Additionally, the model may be misused for unethical data collection or surveillance purposes. It is important to ensure that the model is not used to violate privacy rights or engage in any form of unauthorized data collection.

Furthermore, the model should not be used to discriminate against or harm specific individuals or groups. It is crucial to avoid using the model for activities that perpetuate bias, prejudice, or discrimination.

It is recommended to provide clear guidelines and ethical considerations to users of the model. This includes promoting responsible use, encouraging ethical decision-making, and discouraging any misuse that may harm individuals or society.

Please note that this answer is based on the provided information and may require further input from the sociotechnic team to address potential misuses comprehensively.

### Bias, Risks, and Limitations

The known or foreseeable issues stemming from the model facebook/xlm-v-base are as follows:

1. Scalability issues: The model faces scalability issues due to the computational complexity of the softmax over the entire vocabulary. This can result in increased pre-training times. However, these issues can potentially be addressed by adopting approximation techniques like adaptive softmax and adaptive inputs.

2. Tokenizer compatibility: Integrating the model into 🤗 Transformers requires ensuring that the tokenizer outputs the same ids/subtokens as the `fairseq` tokenizer. A comparison script has been used to verify the tokenizer compatibility.

3. Memory footprint: Scaling the vocabulary can significantly increase the memory footprint of the model. However, it is believed that memory-related issues become less problematic when working with larger models, where the number of non-embedding parameters outweighs the size of the vocabulary embedding matrix.

4. Lack of official integration with `fairseq`: Currently, XLM-V is not officially integrated into the `fairseq` library. However, the model can be loaded with `fairseq` and there is an open merge request for adding the model and usage readme to `fairseq`.

5. Sociotechnical limitations: As a sociotechnic, it is important to consider the potential long-term societal impacts of the model. Foreseeable harms, misunderstandings, and limitations related to ethical implications, bias, and fairness should be thoroughly investigated to ensure responsible deployment and use of the model.

Please note that the information provided is based on the given references, and further details may be needed to provide a comprehensive analysis.

### Recommendations

Based on the provided references, here are the recommendations with respect to the foreseeable issues about the model facebook/xlm-v-base:

1. Scalability Issues: The model faces scalability issues due to increased pre-training times caused by the computational complexity of the softmax over the entire vocabulary. To address this, it is recommended to adopt approximation techniques like adaptive softmax and adaptive inputs.

2. Tokenizer Integration: It is crucial to ensure that the tokenizer in 🤗 Transformers outputs the same ids/subtokens as the `fairseq` tokenizer. This ensures consistency and compatibility with other models and datasets.

3. Memory Footprint: Scaling the vocabulary can significantly increase the memory footprint of the model. However, it is believed that memory-related issues become less problematic when working with larger models, where the number of non-embedding parameters outweighs the size of the vocabulary embedding matrix.

4. Tokenizer Differences: Some sentences may have slightly different output compared to the `fairseq` tokenizer, although this occurs infrequently. It is important to be aware of these differences and assess their impact on downstream tasks.

5. Future Investigations: Further investigation is recommended to explore the Zipf ceiling phenomenon discussed in Section 6 by increasing the vocabulary beyond 2M tokens and utilizing more data. This may help improve the model's performance and coverage.

In summary, it is recommended to address scalability issues through approximation techniques, ensure compatibility with the `fairseq` tokenizer, monitor memory footprint, assess and manage tokenizer differences, and conduct future investigations for continuous improvement of the model.

## Training Details

### Training Data

The training data for the model facebook/xlm-v-base is a multilingual corpus called CC100, which contains 2.5 TB of data split between 116 languages. The dataset is exclusively used for constructing vocabularies and pretraining the models. For more information, please refer to Reference 1.

### Training Procedure

#### Preprocessing

The preprocessing for the data of the model facebook/xlm-v-base involves tokenization and resizing/rewriting.

The tokenization process is performed using the XLM-V tokenizer. This tokenizer has the ability to segment Chinese text into individual entities and separate shared roots from the same word in different languages. It is able to meaningfully break down phrases, producing semantically meaningful tokenizations.

For example, in the phrase "剑桥大学本科生和研究生" (translated as "Cambridge University undergraduates and postgraduates"), the XLM-V tokenizer outputs the tokens "剑桥大学" (Cambridge University), "本科生" (undergraduates), "和" (and), and "研究生" (postgraduates).

The resizing/rewriting process involves constructing a multilingual vocabulary for the model. The authors of the model use a method proposed by Zheng et al. (2021) to build the vocabulary. This method involves training individual sentencepiece models for each language and creating a shared lexicon by taking the union of each language-specific vocabulary. The authors construct a binary vector for each language, representing the lexicon of that language. The binary vectors are then clustered to group lexically similar languages together. Finally, a unified multilingual vocabulary is constructed by combining the per-cluster vocabularies.

The resulting vocabulary has a capacity of one million tokens. Tokenizations using this vocabulary are typically more semantically meaningful and shorter compared to the XLM-R tokenizer. This improved vocabulary allows the XLM-V model to outperform XLM-R on various tasks, including natural language inference, question answering, and named entity recognition.

In summary, the preprocessing for the data of the model facebook/xlm-v-base involves tokenization using the XLM-V tokenizer and resizing/rewriting to construct a multilingual vocabulary. The XLM-V tokenizer is able to segment Chinese text and produce semantically meaningful tokenizations. The resizing/rewriting process involves training individual sentencepiece models, creating a shared lexicon, clustering binary vectors, and constructing a unified multilingual vocabulary with a capacity of one million tokens.

#### Training Hyperparameters

To train the model `facebook/xlm-v-base`, we followed the training procedure described in `XLM-R`. Here are the training hyperparameters:

- Dataset: CC100 dataset
- Sampling temperature: 0.3
- Optimizer: Adam with β1 = 0.9, β2 = 0.98, and ϵ = 1e-6
- Learning rate: 6e-4
- Warmup steps: 15,000
- Batch size: 8,192 (distributed across 256 A100 GPUs)
- Total training iterations: 1.5M
- Maximum sequence length per batch: 512
- Pretraining task: Masked Language Model (MLM)
- Masking rate: 15%

Please note that there is no mention of the code or specific command used for training the `facebook/xlm-v-base` model. For more detailed information, it is recommended to refer to the references provided.

#### Speeds, Sizes, Times

The model card for facebook/xlm-v-base is as follows:

## Model Description

The facebook/xlm-v-base model is a multilingual language model with a 1M token vocabulary. It was developed to overcome the vocabulary bottleneck in multilingual masked language models. The model has been trained on a wide range of tasks, including natural language inference (XNLI), question answering (MLQA, XQuAD, TyDiQA), and named entity recognition (WikiAnn). It outperforms the previous XLM-R model on all tested tasks, especially in low-resource languages.

## Model Information

* Model: facebook/xlm-v-base
* Architecture: Transformer
* Vocabulary Size: 1M tokens
* Trainable Parameters: [More Information Needed]

## Training

The training process for the facebook/xlm-v-base model is documented in the repository. The initial version of the repository was released on 05.02.2023. The model was trained using the XLM-V Integration into 🤗 Transformers. The training steps are also mentioned in the issue link provided.

## Performance

The facebook/xlm-v-base model has shown improved performance compared to the XLM-R model. It achieves better results on tasks such as natural language inference, question answering, and named entity recognition. However, specific performance metrics, such as throughput, start or end time, and checkpoint sizes, are not mentioned in the provided references. [More Information Needed]

## Scalability

While the facebook/xlm-v-base model has strengths in terms of performance, there are scalability issues that need to be addressed. Scaling the vocabulary can increase pre-training times due to the computational complexity of the softmax over the entire vocabulary. However, approximation techniques such as adaptive softmax and adaptive inputs can be adopted to mitigate these issues. Additionally, scaling the vocabulary can also increase the number of trainable parameters. [More Information Needed]

Overall, the facebook/xlm-v-base model is a powerful multilingual language model that overcomes the vocabulary bottleneck and achieves impressive performance on various tasks. Further investigation is needed to gather more information about the model's specific training details, performance metrics, and scalability.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model facebook/xlm-v-base evaluates on the following benchmarks and datasets:

1. XNLI (Cross-lingual Natural Language Inference): This benchmark evaluates whether a premise sentence entails, contradicts, or is neutral toward a hypothesis sentence. It uses crowd-sourced English data translated into 10 other languages for evaluation, and the Multi-Genre Natural Language Inference Corpus (MultiNLI) data for training.

2. MLQA (Multilingual Question Answering): MLQA is a QA evaluation dataset that includes over 12K instances in English and 5K instances in each of the 6 target languages. It is created by mining target language sentences parallel to English sentences from Wikipedia, crowd-sourcing annotations in English, and aligning the question and answer spans.

These benchmarks are used to assess the performance of the facebook/xlm-v-base model.

#### Factors

The foreseeable characteristics that will influence how the model facebook/xlm-v-base behaves include:

1. Domain and Context: The model's performance may vary across different domains and contexts. It is important to evaluate the model's performance in various domains to understand its strengths and limitations.

2. Population Subgroups: The model may display disparities in performance across different population subgroups. Evaluating the model's performance disaggregated across factors such as language, geography, cultural context, or demographic characteristics can help uncover potential biases or disparities in its behavior.

To obtain a detailed understanding of these characteristics, it is crucial to conduct thorough evaluations and analyses that consider the above factors. This includes evaluating the model's performance on specific datasets like MLQA, XNLI, CC100, and FLoRes-200, as mentioned in the references. Additionally, conducting further research and gathering data on the model's performance in different domains and population subgroups will provide a more comprehensive understanding of its behavior.

#### Metrics

The metrics used for evaluation in light of trade-offs between different errors for the model facebook/xlm-v-base are not explicitly mentioned in the provided references. [More Information Needed]

### Results

The evaluation results of the model facebook/xlm-v-base are not explicitly mentioned in the provided references. Hence, we need more information to provide the evaluation results based on the Factors and Metrics.

#### Summary

The evaluation results for the model facebook/xlm-v-base show that it outperforms all baselines on XNLI, including XLM-R (1M), by 1.34% in terms of absolute accuracy. It also outperforms the model by Chung et al. (2020) by 1.11% in absolute accuracy. The model is trained for 300,000 iterations with a batch size of 2,048 on the CC100 corpus. The evaluation results indicate that XLM-V is particularly effective in low-resource languages and performs well on tasks such as natural language inference, question answering, and named entity recognition. The model's vocabulary is designed to provide sufficient coverage for each individual language, resulting in semantically meaningful tokenizations and shorter average sequence lengths compared to XLM-R. The model serves as an efficient form of conditional compute. However, further investigation is needed to explore increasing the vocabulary beyond 2M tokens and using more data to address the Zipf ceiling.

## Model Examination

The model card for the model facebook/xlm-v-base:

### Model Description

The facebook/xlm-v-base is a multilingual language model with a 1 million token vocabulary. It outperforms the XLM-R model on various tasks, including natural language inference (XNLI), question answering (MLQA, XQuAD, TyDiQA), and named entity recognition (WikiAnn). The model has been trained using a method that constructs multilingual vocabularies, which involves training individual monolingual sentencepiece models for each language in the dataset and clustering the lexical representation vectors using K-Means. The final vocabulary consists of 901,629 tokens, with an average of over 90% of the tokens in each cluster being unique.

### Experimental Section on Explainability/Interpretability

[More Information Needed]

## Environmental Impact

- **Hardware Type:** The hardware type that the model facebook/xlm-v-base is trained on is the A100 GPU.
- **Software Type:** The model `facebook/xlm-v-base` is trained on the Unigram Language Model (ULM) algorithm, which is used for subword segmentation and vocabulary construction. The model utilizes a multilingual vocabulary that is constructed by training individual sentencepiece models for each language and then creating a shared lexicon by taking the union of these language-specific vocabularies. The model also incorporates the average log probability (ALP) to evaluate the vocabulary capacity for individual languages in the multilingual vocabulary. The model has been integrated into the 🤗 Transformers library and follows the same tokenization process as the `fairseq` tokenizer. It has been uploaded to the 🤗 Transformers Model Hub and is available under the name `facebook/xlm-v-base`.
- **Hours used:** The model facebook/xlm-v-base was pretrained using the CC100 dataset for a total of 1.5 million iterations. The training procedure followed the same approach as XLM-R and utilized a sampling temperature of 0.3 to increase exposure to low- and medium-resource language examples. The Adam optimizer with default parameters (β1=0.9, β2=0.98, ϵ=1e-6) was used, along with a learning rate of 6e-4. The model was pretrained on examples with a maximum sequence length of 512 and a batch size of 8,192 distributed across 256 A100 GPUs. The Masked Language Model (MLM) task with a 15% masking rate was used for training.

The exact amount of time used to train the model is not provided in the references. [More Information Needed]
- **Cloud Provider:** The cloud provider on which the model facebook/xlm-v-base is trained is not specified in the given references. [More Information Needed]
- **Carbon Emitted:** Based on the provided information, there is no direct mention of the amount of carbon emitted when training the model facebook/xlm-v-base. Therefore, [More Information Needed].
## Technical Specification

### Model Architecture and Objective

The model architecture of facebook/xlm-v-base is described in the paper "XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models" by Davis Liang et al. It is a multilingual language model with a one million token vocabulary trained on 2.5TB of data from Common Crawl, similar to XLM-R.

The objective of XLM-V is to address the vocabulary bottleneck in multilingual models like XLM-R. It achieves this by reducing over-tokenization for low-resource languages and assigning vocabulary capacity to achieve sufficient coverage for each individual language. The tokenizations produced by XLM-V's vocabulary are typically more semantically meaningful and shorter compared to XLM-R.

XLM-V outperforms XLM-R on various tasks, including natural language inference (XNLI), question answering (MLQA, XQuAD, TyDiQA), and named entity recognition (WikiAnn).

In terms of implementation details, the tokenizer in 🤗 Transformers for XLM-V should output the same ids/subtokens as the `fairseq` tokenizer. To ensure this, a script called `xlm_v_tokenizer_comparison.py` is used to load all 176 languages from the WikiANN dataset, tokenize each sentence, and compare the tokenization results.

For more information about model baselines and specific details, please refer to the XLM-V paper.

### Compute Infrastructure

The compute infrastructure for the model facebook/xlm-v-base is as follows:

- The model was trained using the CC100 dataset.
- The training was done with a sampling temperature of 0.3 to increase the amount of low- and medium-resource language examples seen during training.
- The Adam optimizer was used with default parameters (β1=0.9, β2=0.98, ϵ=1e-6).
- The learning rate used was 6e-4.
- A warmup of 15,000 steps was used.
- The training was done with a batch size of 8,192 distributed across 256 A100 GPUs.
- The total training time is not mentioned.
- The model was pretrained using the Masked Language Model (MLM) task with a masking rate of 15%.
- The pretraining was performed for 1.5M iterations.
- Each batch consists of examples concatenated up to the maximum sequence length of 512.

Unfortunately, the specific compute infrastructure details such as CPU, RAM, and GPU specifications are not mentioned in the provided references. [More Information Needed]

## Citation

```
@misc{davis-xlmv,
    author = {Davis Liang and
              Hila Gonen and
              Yuning Mao and
              Rui Hou and
              Naman Goyal and
              Marjan Ghazvininejad and
              Luke Zettlemoyer and
              Madian Khabsa and
              Meta Ai},
    title  = {XLM-V: Overcoming the Vocabulary Bottleneck in Multilingual Masked Language Models},
    url    = {https://arxiv.org/pdf/2301.10472.pdf}
}
```

