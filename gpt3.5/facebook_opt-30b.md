# Model Card for facebook/opt-30b

The model facebook/opt-30b is a large decoder-only transformer language model developed by Meta AI, released on May 3, 2022, with a primary focus on research into Language Models, especially as it pertains to Responsible AI. It is designed for use by academic researchers and the related research community, and it is not intended for production use or real-world deployments. [More Information Needed]

## Model Details

### Model Description

Model name: facebook/opt-30b

Description:
OPT-30B is a large decoder-only transformer language model developed by Meta AI. It was released on May 3, 2022, and the version described in this model card is 1.0.0.

Model architecture:
OPT-30B follows a decoder-only transformer architecture. More architectural details can be found in Table 1 of the referenced paper.

Training procedures:
OPT-30B was trained using the AdamW optimizer with (β1, β2) set to (0.9, 0.95) and a weight decay of 0.1. The model was trained with a linear learning rate schedule, starting from 0 and gradually increasing to the maximum learning rate over the first 2000 steps or 375M tokens for smaller baselines. The learning rate then decayed down to 10% of the maximum LR over 300B tokens. Batch sizes ranged from 0.5M to 4M depending on the model size.

Parameters:
OPT-30B has a parameter size of 30 billion.

Important disclaimers:
- OPT-30B and other smaller baseline models are made available through a non-commercial use license agreement provided in the model license.
- Loss divergences were observed during training, and the model was restarted from earlier checkpoints with a healthy dynamic loss scalar and decreasing activation norms to recover and continue training.
- More information about training data, training process, and other applied approaches can be found in the referenced paper and the Meta AI Research Blog.
- For more details about OPT-30B, refer to the paper and the metaseq open-source repository.

Please note that additional information may be needed to provide a more comprehensive model card for facebook/opt-30b.

- **Developed by:** Susan Zhang; Stephen Roller; Naman Goyal; Mikel Artetxe; Moya Chen; Shuohui Chen; Christopher Dewan; Mona Diab; Xian Li; Victoria Lin; Todor Mihaylov; Myle Ott; Sam Shleifer; Kurt Shuster; Daniel Simig; Singh Koura; Anjali Sridhar; Tianlu Wang; Luke Zettlemoyer
- **Funded by:** The people or organizations that fund the project of the model facebook/opt-30b are:
- [More Information Needed]
- **Shared by:** The contributors who made the model facebook/opt-30b available online as a GitHub repo are Bashlykov, Binh Tang, Uriel Singer, Yuchen Zhang, Armen Aghajanya, Lili Yu, and Adam Polyak.
- **Model type:** The facebook/opt-30b model is a Transformer language model with 30 billion parameters, trained using pre-training data from a union of five datasets, and it outperforms other models on various tasks according to the results in Table 2.
- **Language(s):** The model facebook/opt-30b uses or processes natural human language from a pre-training dataset that includes a union of multiple datasets, including RoBERTa datasets, a subset of the Pile, and the Pushshift.io Reddit dataset.
- **License:** The license being used for the model facebook/opt-30b is the OPT-175B non-commercial use license agreement. For more information, please refer to the model license provided in our model license.
- **Finetuned from model:** [More Information Needed]
### Model Sources

- **Repository:** https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling
- **Paper:** https://arxiv.org/pdf/2205.01068.pdf
- **Demo:** Unfortunately, the provided references don't contain any information about the demo of the model facebook/opt-30b. Therefore, I don't have the link to the demo of the model at the moment.
## Uses

### Direct Use

The model facebook/opt-30b can be used without fine-tuning, post-processing, or plugging into a pipeline. It is a pre-trained language model that can generate text based on given input prompts. 

To use the model, you can simply input a text prompt and generate text using the `generate` method provided by the Hugging Face library. Here is an example code snippet:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained model
model_name = "facebook/opt-30b"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the model to generate mode
model.eval()

# Input prompt
prompt = "Once upon a time"

# Tokenize the input
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate text
output = model.generate(input_ids, max_length=100)

# Decode the generated output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Print the generated text
print(generated_text)
```

Please note that this is a simplified code snippet and you may need to adjust it based on your specific use case.

### Downstream Use

The model facebook/opt-30b can be fine-tuned for various downstream tasks or integrated into larger ecosystems or apps. When fine-tuned, it can be used for tasks such as text classification, named entity recognition, sentiment analysis, question answering, machine translation, and more.

To fine-tune the model, you can use the Hugging Face library and follow the standard fine-tuning procedure. Here is a code snippet to illustrate the process:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b")
model = AutoModelForSequenceClassification.from_pretrained("facebook/opt-30b")

# Prepare your training data
train_texts = ["text example 1", "text example 2"]
train_labels = [0, 1]

# Tokenize the training data
train_encodings = tokenizer(train_texts, truncation=True, padding=True)

# Fine-tune the model
model.train_model(train_encodings, train_labels)
```

After fine-tuning, you can use the model to make predictions on new input texts by encoding them and passing them through the model:

```python
# Prepare your test data
test_texts = ["test example 1", "test example 2"]

# Tokenize the test data
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# Get the model predictions
outputs = model(**test_encodings)
predictions = outputs.logits.argmax(dim=-1)

print(predictions)
```

Note that this code snippet is a simplified example, and you may need to adapt it to your specific task and dataset. Additionally, fine-tuning requires labeled data for the target task.

[More Information Needed]

### Out-of-Scope Use

The model facebook/opt-30b is not mentioned in the provided references. Therefore, I don't have the necessary information to answer the question about how the model may foreseeably be misused and address what users ought not do with it.

### Bias, Risks, and Limitations

The known or foreseeable issues stemming from the model `facebook/opt-30b` include:

1. Bias and Safety Limitations: Similar to other large language models, `facebook/opt-30b` has limitations in terms of bias and safety. The diversity of the training data can impact the quality of the model. There can also be issues with generation diversity and hallucination. The model is not immune to the various issues that affect modern large language models.

2. Quality Issues: `facebook/opt-30b` may have quality issues in terms of generating factually incorrect statements. This can be particularly problematic in domains where information accuracy is critical, such as healthcare and scientific discovery.

3. Toxic Language and Stereotype Reinforcement: The model has a high propensity to generate toxic language and reinforce harmful stereotypes, even when provided with innocuous prompts. Adversarial prompts that trigger toxicity are also easily found. Mitigations for toxicity and biases have been explored, but they are not applied in the first release of the model.

4. Lack of Complete Evaluation: While extensive evaluations have been performed on `facebook/opt-30b` for standard evaluation datasets and safety, bias, and inclusion evaluations, these evaluations may not fully capture all the limitations of the model. The qualitative observations suggest that the model suffers from various limitations.

5. Sociotechnical Considerations: Deploying language models like `facebook/opt-30b` at scale raises ethical and social risks. Responsible AI research is encouraged, and transparency around model development is provided to increase accountability. However, the model's access is limited to reduce the environmental impact and potential risks associated with large-scale research.

In summary, `facebook/opt-30b` has limitations related to bias, safety, generation quality, factual correctness, toxicity, and reinforcement of harmful stereotypes. Evaluations have been conducted, but they may not cover all limitations. Sociotechnical considerations are addressed by emphasizing responsible AI research and increasing transparency. Further information on specific mitigation approaches and code blocks is needed.

### Recommendations

Based on the references provided, the recommendations with respect to the foreseeable issues about the model facebook/opt-30b are as follows:

1. Increase Communication and Transparency: Releasing the model with a non-commercial license aims to increase communication and transparency regarding the limitations, bias, safety, and quality issues of the model. This is in line with the recommendations for individual researchers generated by the Partnership for AI.

2. Full Accountability: The development lifecycle of the model is fully accountable, and all details involved in training OPT-175B are disclosed through logbooks, code, and providing researchers access to model weights. This aligns with the governance guidance outlined by NIST.

3. Responsible AI Research: The model is released to enable Responsible AI research into Language Models. The primary intended users are researchers and the related research community.

4. Environmental Impact: Providing researchers with direct access to the parameters of the model aims to reduce the environmental impact of pursuing research at this scale.

5. Ethical and Social Risks: There is a growing body of work highlighting ethical and social risks from deploying language models with emergent capabilities at scale. By limiting access to the model, the aim is to prevent deployment in areas that may not be aligned with commercial interests.

6. Detailed Evaluation: Extensive evaluation has been carried out for the released models, including performance evaluation on standard datasets, safety, bias, and inclusion evaluations. However, it is noted that such evaluations may not fully capture all limitations of the model.

Overall, the recommendations focus on increasing transparency, accountability, and responsible use of the model, while also addressing potential issues related to bias, safety, and environmental impact.

## Training Details

### Training Data

The training data for the model facebook/opt-30b consists of a concatenation of datasets used in RoBERTa, the Pile, and PushShift.io Reddit. The datasets were filtered to contain predominantly English text, with some non-English data from CommonCrawl still present. Further details on data selection and preprocessing can be found in the Data Card (Appendix C) of the model's documentation.

### Training Procedure

#### Preprocessing

The provided references do not contain specific information about the preprocessing steps for the data used in the model facebook/opt-30b. Therefore, I cannot provide the details about tokenization, resizing/rewriting, or any other preprocessing steps for the data. [More Information Needed]

#### Training Hyperparameters

The training hyperparameters for the model facebook/opt-30b are as follows:

1. Optimizer: AdamW optimizer with β1 = 0.9, β2 = 0.95.
2. Weight Decay: 0.1.
3. Learning Rate Schedule: Linear learning rate schedule with warm-up. The learning rate starts from 0 and increases to the maximum learning rate over the first 2000 steps.
4. Learning Rate Decay: The learning rate decays down to 10% of the maximum learning rate over 300B tokens.
5. Gradient Clipping: Gradient clipping was lowered from 1.0 to 0.3 for stability.
6. Batch Sizes: Batch sizes range from 0.5M to 4M depending on the model size.
7. Weight Initialization: Normal distribution with zero mean and standard deviation of 0.006. Standard deviation for output layers are scaled by a 1.0/ √ 2L term, where L is the total number of layers.
8. Bias Initialization: All bias terms are initialized as 0.
9. Activation Function: ReLU activation.
10. Sequence Length: Sequence length is set to 2048.

Please note that this information is based on the provided references, and if there are any additional specific details required, please let me know.

#### Speeds, Sizes, Times

The model card for facebook/opt-30b provides the following details:

Throughput: [More Information Needed]

Start or end time: [More Information Needed]

Checkpoint sizes: [More Information Needed]

Unfortunately, the provided references do not contain specific information about the throughput, start or end time, or checkpoint sizes for the facebook/opt-30b model. Additional information is needed to provide these details.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model facebook/opt-30b evaluates on the following benchmarks and datasets:

1. Hate speech detection, stereotype awareness, and toxic content generation benchmarks (Blodgett et al., 2021; Jacobs and Wallach, 2021).
2. Dialogue Safety evaluations, including SaferDialogues (Ung et al., 2021) and Safety Bench Unit Tests (Xu et al., 2020).
3. ETHOS dataset for identifying racist or sexist statements (Mollas et al., 2020; Chiu and Alexander, 2021).
4. ConvAI2 dataset for dialogue tasks (Xu et al., 2021b).
5. Wizard-of-Internet dataset for unsupervised tasks (not specified, but mentioned in reference 6).

Please note that the model card does not provide specific details about the results or performance of the model on these benchmarks and datasets.

#### Factors

The foreseeable characteristics that will influence how the model `facebook/opt-30b` behaves include:

1. **Domain and Context**: The model's behavior will be influenced by the domain and context in which it is used. For example, if the model is used in social media discussions or dialogue systems, it may be more prone to exhibiting stereotypical biases and generating toxic content.

2. **Population Subgroups**: The model's performance may vary across different population subgroups. Disaggregated evaluation across factors such as race, gender, and ethnicity is important to uncover disparities in performance. This helps identify potential bias and ensure fairness and inclusivity.

3. **Bias and Safety**: The model has limitations in terms of bias and safety, similar to other large language models. It is important to evaluate and address potential biases in the model's output, particularly related to hate speech detection, stereotype awareness, and toxic content generation.

4. **Evaluation and Data Selection**: Evaluation of the model should be performed using benchmarks that are designed to assess its performance in terms of safety, fairness, and responsibility. The choice of evaluation data should be carefully selected to ensure comparability with similar scale models in the literature.

In conclusion, the behavior of the `facebook/opt-30b` model can be influenced by factors such as domain and context, population subgroups, biases, safety, and the selection of evaluation data. Disaggregated evaluation is crucial to uncover disparities in performance and address potential biases.

#### Metrics

Based on the references provided, the information about the metrics used for evaluation in light of tradeoffs between different errors for the model facebook/opt-30b is not explicitly mentioned. Therefore, [More Information is Needed] to answer this question.

### Results

Based on the given references, the evaluation results of the model facebook/opt-30b are not mentioned. Therefore, [More Information Needed] to provide the evaluation results based on the Factors and Metrics.

#### Summary

The evaluation results for the model facebook/opt-30b, also known as OPT-175B, show that it performs considerably better than Davinci in various one-shot through few-shot configurations. This improvement may be due to the additional safety control mechanisms in the Davinci API and the presence of unmoderated social media discussions in the pre-training dataset, which provide additional inductive bias for classification tasks. OPT-175B also performs well on dialogue safety evaluations, demonstrating its ability to recover from explicit safety failures. Furthermore, OPT-175B shows lower toxicity compared to models fine-tuned on curated dialogue datasets. It has been evaluated on benchmarks related to hate speech detection, stereotype awareness, and toxic content generation, providing insights into its limitations. OPT-175B has been tested on the ETHOS dataset, demonstrating its ability to identify whether English statements are racist, sexist, or neither. The model outperforms unsupervised models on various tasks and performs competitively with fully supervised models. However, there may be some limitations in terms of perplexity and UF1 scores on specific datasets. Overall, OPT-175B is intended for research purposes and not for production use or real-world deployments.

## Model Examination

The model card for the facebook/opt-30b model does not provide any specific information about work on explainability/interpretability. Therefore, I am not able to provide any details or code related to this aspect of the model. [More Information Needed]

## Environmental Impact

- **Hardware Type:** Based on the given references, there is no specific information available about the hardware type on which the model facebook/opt-30b is trained. [More Information Needed]
- **Software Type:** The model facebook/opt-30b is trained using the AdamW optimizer with (β1, β2) set to (0.9, 0.95) and a weight decay of 0.1. It follows a linear learning rate schedule, warming up from 0 to the maximum learning rate over the first 2000 steps and decaying down to 10% of the maximum LR over 300 billion tokens. The batch sizes range from 0.5 million to 4 million depending on the model size. The weight initialization follows the settings provided in the Megatron-LM codebase, using a normal distribution with zero mean and a standard deviation of 0.006. The models are trained with ReLU activation and a sequence length of 2048. Dropout with a rate of 0.1 is applied throughout, except for embeddings. The gradient norms are clipped at 1.0, and there are midflight changes that reduce the norm threshold down to 0.3. The dataset used for training the OPT-175B model is a union of five datasets, including those used by RoBERTa and the Pushshift.io Reddit dataset. The primary intended use of the OPT-175B model is for research into Language Models, especially as it pertains to Responsible AI. It is not released for production use or real-world deployments.
- **Hours used:** The amount of time used to train the model `facebook/opt-30b` is not mentioned in the provided references. [More Information Needed]
- **Cloud Provider:** The cloud provider on which the model facebook/opt-30b is trained is not mentioned in the provided references. [More Information Needed]
- **Carbon Emitted:** Based on the provided references, there is no information available about the amount of carbon emitted specifically when training the model facebook/opt-30b. Therefore, I am unable to provide the carbon emissions for this particular model. [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The model architecture and objective for the model facebook/opt-30b are not provided in the given references. [More Information Needed]

### Compute Infrastructure

The compute infrastructure for the model facebook/opt-30b is not provided in the given references. [More Information Needed]

## Citation

```
@misc{susan-opt,
    author = {Susan Zhang and
              Stephen Roller and
              Naman Goyal and
              Mikel Artetxe and
              Moya Chen and
              Shuohui Chen and
              Christopher Dewan and
              Mona Diab and
              Xian Li and
              Victoria Lin and
              Todor Mihaylov and
              Myle Ott and
              Sam Shleifer and
              Kurt Shuster and
              Daniel Simig and
              Singh Koura and
              Anjali Sridhar and
              Tianlu Wang and
              Luke Zettlemoyer},
    title  = {OPT: Open Pre-trained Transformer Language Models},
    url    = {https://arxiv.org/pdf/2205.01068.pdf}
}
```

