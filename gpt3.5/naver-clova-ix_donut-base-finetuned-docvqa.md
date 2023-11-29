# Model Card for naver-clova-ix/donut-base-finetuned-docvqa

The model naver-clova-ix/donut-base-finetuned-docvqa is a Visual Document Understanding (VDU) based model that has been fine-tuned for the DocVQA (Document Visual Question Answering) task. It predicts answers for questions related to document images by capturing both visual and textual information within the image. The model shows stable performance regardless of the size of datasets and complexity of the tasks, making it suitable for industry use. For more information, refer to the provided references.

## Model Details

### Model Description

Model Name: naver-clova-ix/donut-base-finetuned-docvqa

Description:
The naver-clova-ix/donut-base-finetuned-docvqa model is an end-to-end VDU (Visual Document Understanding) model designed for general understanding of document images. It is based on the Donut architecture, which consists of a Transformer-based visual encoder and a textual decoder. The model does not rely on OCR functionality but instead uses a visual encoder to extract features from document images. The derived features are then mapped into a sequence of subword tokens by the textual decoder to construct a structured output.

Architecture:
The visual encoder of the model converts the input document image into a set of embeddings. In this particular implementation, the Swin Transformer is used as the encoder network, as it has shown the best performance in document parsing during preliminary studies.

Training Procedures:
The model is trained as a visual language model over visual corpora, specifically document images. The objective is to minimize the cross-entropy loss of next token prediction, conditioned on both the image and previous contexts. The training task can be interpreted as a pseudo-OCR task, where the model learns to read all texts in the image in reading order (from top-left to bottom-right).

Decoder:
The textual decoder generates a token sequence based on the derived features from the visual encoder. The decoder architecture used in this model is BART (Bidirectional and Auto-Regressive Transformer), with the decoder model weights initialized from a publicly available pre-trained multi-lingual BART model.

Parameters:
The model parameters include the dimension of the latent vectors of the encoder, the size of the token vocabulary, and the hyperparameter m which determines the length of the token sequence generated by the decoder.

Important Disclaimers:
1. The model does not rely on OCR functionality and is designed for OCR-free document understanding.
2. The model's performance has been evaluated on the IIT-CDIP dataset, which is a set of 11M scanned English document images. Synthetic documents generated using the SynthDoG tool and Wikipedia articles in Chinese, Japanese, Korean, and English were also used for training. Performance on languages other than English may vary.
3. The model's performance remains stable regardless of the size of datasets and complexity of tasks, as shown in the evaluation results.

Please note that this is a high-level summary of the model card description. For more detailed information, please refer to the provided references.

- **Developed by:** Geewook Kim;  Hong; Moonbin Yim; Jeongyeon Nam; Jinyoung Park; Jinyeong Yim; Wonseok Hwang; Sangdoo Yun; Dongyoon Han; Seunghyun Park; Naver Clova; Naver Search; Naver Ai Lab;  Tmax;  Google;  Lbox
- **Funded by:** The people or organizations that fund the project of the model naver-clova-ix/donut-base-finetuned-docvqa are:
- Naver Clova
- **Shared by:** The contributors who made the model naver-clova-ix/donut-base-finetuned-docvqa available online as a GitHub repo are not mentioned in the provided references. [More Information Needed]
- **Model type:** The model naver-clova-ix/donut-base-finetuned-docvqa is a visual language model that is trained as a pseudo-OCR task on document images using a commercial CLOVA OCR API and synthetic data generation, and is fine-tuned for document visual question answering (DocVQA) by jointly conditioning on the image and previous contexts.
- **Language(s):** The model naver-clova-ix/donut-base-finetuned-docvqa processes document images and their text annotations for the task of document understanding and visual question answering (DocVQA), using a Transformer-based visual encoder and textual decoder modules.
- **License:** The model naver-clova-ix/donut-base-finetuned-docvqa is published under the MIT license. You can find the license and its terms at the following link: [MIT license](https://github.com/clovaai/donut/blob/official/LICENSE).


- **Finetuned from model:** The model naver-clova-ix/donut-base-finetuned-docvqa is fine-tuned from the base model `donut-base`. The link to the base model is [here](https://huggingface.co/naver-clova-ix/donut-base/tree/official).
### Model Sources

- **Repository:** https://github.com/clovaai/donut
- **Paper:** https://arxiv.org/pdf/2111.15664.pdf
- **Demo:** The link to the demo of the model naver-clova-ix/donut-base-finetuned-docvqa is available in the references as follows:

[gradio space web demo](https://huggingface.co/spaces/nielsr/donut-docvqa)
## Uses

### Direct Use

Model: naver-clova-ix/donut-base-finetuned-docvqa

Description:
The naver-clova-ix/donut-base-finetuned-docvqa model is designed for the task of Visual Document Understanding (VDU), specifically for Document Visual Question Answering (DocVQA). It is capable of extracting useful information from document images by capturing both visual and textual information within the image. 

To use the model without fine-tuning, post-processing, or plugging into a pipeline, you can follow these steps:

1. Install the required dependencies:
```python
!pip install transformers torch
```

2. Load the model and tokenizer:
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "naver-clova-ix/donut-base-finetuned-docvqa"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

3. Preprocess the document image and question pair:
```python
def preprocess(image_path, question):
    # Load and preprocess the document image
    document_image = preprocess_image(image_path)
    
    # Tokenize the question
    question_tokens = tokenizer.encode(question, add_special_tokens=False, return_tensors="pt")
    
    # Prepare inputs for the model
    inputs = {
        "input_ids": question_tokens,
        "attention_mask": question_tokens.new_ones(question_tokens.shape),
        "visual_inputs": [document_image],
    }
    
    return inputs
```

4. Generate the answer for a given document image and question:
```python
def generate_answer(image_path, question):
    # Preprocess inputs
    inputs = preprocess(image_path, question)
    
    # Generate answer
    output = model.generate(**inputs)
    
    # Decode the generated answer
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return answer
```

5. Use the model to generate answers:
```python
image_path = "path/to/document/image.jpg"
question = "What is the total price of the receipt?"

answer = generate_answer(image_path, question)
print("Answer:", answer)
```

Please note that the `preprocess_image` function and the specific preprocessing steps for the document image are not provided in the references, so you would need to implement or obtain them separately. Additionally, the exact details of the visualization and textual information extraction mechanisms of the model are not mentioned, hence more information is needed for a complete understanding of the model's inner workings.

### Downstream Use

The model naver-clova-ix/donut-base-finetuned-docvqa can be used when fine-tuned for a task or when plugged into a larger ecosystem or app. It is based on the Donut model, which is pre-trained to read texts in document images and understand the document image. The model takes a document image as input and generates a token sequence that can be converted into a JSON representation of the desired output information.

To use the model for a specific task, you can fine-tune it on a labeled dataset related to that task. For example, in document classification, the decoder can be trained to generate a token sequence like "[START class][memo][END class]", which corresponds to the JSON {"class": "memo"}. Special tokens are introduced to represent different classes.

When plugged into a larger ecosystem or app, the model can be used to extract useful information from document images. It can be integrated into applications for various tasks such as document classification, information extraction, and visual question answering. The model's stable performance and flexibility make it suitable for different datasets and tasks.

Here is a code snippet that demonstrates how to use the model for inference:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = AutoModelForSeq2SeqLM.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

# Example document image
document_image = "path/to/document_image.jpg"

# Tokenize the document image
inputs = tokenizer(document_image, return_tensors="pt")

# Generate token sequence
outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

# Decode the token sequence
decoded_tokens = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Convert token sequence to JSON
json_output = convert_to_json(decoded_tokens)

# Use the JSON output for further processing or display
print(json_output)
```

Please note that the code snippet provided is for illustration purposes and may need to be adapted to your specific use case.

### Out-of-Scope Use

Based on the provided references, it is not explicitly mentioned how the model naver-clova-ix/donut-base-finetuned-docvqa may be misused or what users ought not to do with the model. Therefore, [More Information Needed] to answer this question accurately.

### Bias, Risks, and Limitations

The known or foreseeable issues stemming from the model naver-clova-ix/donut-base-finetuned-docvqa are as follows:

1. OCR Dependency: The conventional OCR-based models rely on external OCR engines, which introduces potential issues such as OCR errors and limited language or document type support. Donut aims to address these issues by being an OCR-free VDU model. However, the model card does not provide specific details on how Donut mitigates these problems.

2. OCR Error Propagation: OCR errors can propagate to the subsequent processes, impacting the overall accuracy and reliability of the model's predictions. Donut takes a different approach by not relying on OCR correction or encoder-decoder architecture on OCR outputs. However, the model card does not elaborate on how Donut overcomes OCR error propagation.

3. Technical Limitations: The model card mentions that Donut achieves competitive scores on the DocVQA dataset, including robustness to handwritten documents. However, it does not provide information about potential limitations or failure cases of the model. Further analysis and evaluation are required to assess its performance in different scenarios.

4. Sociotechnical Limitations: As a sociotechnic, it is important to consider potential ethical implications and societal impacts of deploying the model. The model card does not provide explicit information on these aspects, such as fairness, bias, or privacy concerns. A thorough sociotechnical analysis should be conducted to identify and address any such limitations.

Please note that the provided references do not offer comprehensive information on all aspects of the model, so some details may be missing. Further investigation and analysis are needed to fully understand the model's limitations and potential issues.

### Recommendations

Based on the provided references, here are some recommendations regarding the foreseeable issues about the model naver-clova-ix/donut-base-finetuned-docvqa:

1. Input Resolution: Increasing the input size can improve the performance of Donut, especially in tasks like DocVQA with larger images and tiny texts. However, it is important to consider the computational costs associated with larger input sizes. Exploring efficient attention mechanisms could be beneficial in reducing computational requirements.

2. OCR System: The performance of OCR-based methods heavily depends on the off-the-shelf OCR engine used. It is recommended to carefully select and evaluate OCR engines based on factors such as speed and accuracy. More information about the specific OCR engines used in the experiments is available in Appendix A.1.

3. Low Resourced Situation: Donut shows robust performance even in low-resource situations where the size of the training set is limited. This is valuable as the target tasks are actively used in industries. However, further analysis and details are needed to understand the specific performance characteristics and limitations in such scenarios.

4. Text Localization: Donut exhibits meaningful results in text localization, as observed through cross-attention maps. These maps can serve as auxiliary indicators, providing insights into how the model attends to desired locations in the input image.

5. IE Task and DocVQA: Real images are crucial for tasks like DocVQA, particularly due to the similarity in image distributions between datasets like IIT-CDIP and DocVQA. Understanding the impact of image realism and distribution on model performance is essential.

6. Performance Stability: Donut demonstrates stable performance across different dataset sizes and task complexities compared to other baselines. This stability is significant considering the active use of the target tasks in industries. Further analysis and details in Section 3.4 provide insights into the model's robustness.

It is important to note that additional information and analyses are needed to provide more comprehensive recommendations regarding the foreseeable issues of the model naver-clova-ix/donut-base-finetuned-docvqa.

## Training Details

### Training Data

The training data of the model naver-clova-ix/donut-base-finetuned-docvqa is not explicitly mentioned in the provided references. However, it is likely that the model was fine-tuned on the DocVQA dataset mentioned in reference 4, which consists of document images and corresponding question-answer pairs. For more information on the data pre-processing or additional filtering, please refer to the documentation provided at https://github.com/clovaai/donut.

### Training Procedure

#### Preprocessing

The preprocessing steps for the data of the model `naver-clova-ix/donut-base-finetuned-docvqa` include tokenization, resizing, and rewriting.

Tokenization:
The input text is tokenized into subword tokens using the BART model. The model uses a vocabulary of size `v`, and each token is represented as a one-hot vector `y_i` where `i` is the index of the token.

Resizing/Rewriting:
The output token sequence is converted to a structured format using a JSON format. Two special tokens, `[START *]` and `[END *]`, are added to the token sequence, where `*` represents each field to extract. If the output token sequence is incorrectly structured, the missing field is assumed to be lost.

Overall, the model is trained to read all texts in the image in reading order, from top-left to bottom-right. The objective is to minimize the cross-entropy loss of next token prediction by conditioning on the image and previous contexts. This task can be interpreted as a pseudo-OCR task.

The model takes a teacher-forcing scheme during training, where the ground truth is used as input instead of the model output from the previous time step. During the test phase, the model generates a token sequence given a prompt.

The preprocessing steps for the data of the model `naver-clova-ix/donut-base-finetuned-docvqa` involve tokenization using the BART model, converting the output token sequence to a JSON format, and following a teacher-forcing scheme during training.

Please note that the specific implementation details or code snippets are not provided in the references, so further information may be needed to get a complete understanding of the preprocessing steps.

#### Training Hyperparameters

To train the model `naver-clova-ix/donut-base-finetuned-docvqa`, the following hyperparameters were used:

- Training optimization algorithm: Adam optimizer [30]
- Training precision: Half-precision (fp16)
- Learning rate: Decreased as the training progresses, with an initial learning rate of 1e-4 for pre-training and a range of 1e-5 to 1e-4 for fine-tuning
- Training steps: Pre-training for 200K steps
- GPUs used: 64 NVIDIA A100 GPUs
- Mini-batch size: 196
- Gradient clipping: Applied with a maximum gradient norm selected from 0.05 to 1.0

Unfortunately, the specific values for the learning rate, gradient clipping, and training steps during fine-tuning are not provided in the references.

#### Speeds, Sizes, Times

The model card description for the model naver-clova-ix/donut-base-finetuned-docvqa is as follows:

The naver-clova-ix/donut-base-finetuned-docvqa model is an OCR-free VDU (Visual Document Understanding) model that achieves state-of-the-art performance in terms of both speed and accuracy on various VDU tasks and datasets. It is a simple and efficient model that does not rely on external OCR engines.

The model has been evaluated on different VDU tasks, including Document Classification and Document Visual Question Answering (DocVQA). In Document Classification tasks, the model outperforms other general-purpose VDU models such as LayoutLM and LayoutLMv2 in terms of accuracy, while using fewer parameters and being 2x faster. It achieves state-of-the-art performance even without relying on additional resources like off-the-shelf OCR engines.

In DocVQA tasks, the model demonstrates its strong understanding ability by capturing both visual and textual information within the document image to predict answers for given questions. It shows competitive performance compared to baselines that depend on external OCR engines, and it is especially robust in processing handwritten documents, known to be challenging.

The model's performance remains stable regardless of the size of datasets and complexity of tasks, making it suitable for low-resource situations. It performs well on both public and private in-service datasets and can extract key information and predict complex structures among the field information.

The model achieves these results with a relatively small number of parameters compared to recent OCR-based models, making it more efficient and cost-effective. The model's code, trained model, and synthetic data generator are available on GitHub.

Regarding throughput, start or end time, checkpoint sizes, and other specific details about the model naver-clova-ix/donut-base-finetuned-docvqa, unfortunately, the provided references do not contain explicit information about these metrics. [More Information Needed]

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model naver-clova-ix/donut-base-finetuned-docvqa evaluates on the following benchmarks or datasets:

1. Document Visual Question Answering (DocVQA) dataset: The model's performance on the DocVQA dataset is reported in the references. It achieves competitive scores with baselines that depend on external OCR engines and shows robustness to handwritten documents.

2. Document Classification: The model's performance on a document classification task is evaluated and reported. It shows state-of-the-art performance among general-purpose VDU models like LayoutLM and LayoutLMv2, while using fewer parameters and being 2x faster.

3. Document Information Extraction: The model is tested on various real document images, including both public benchmarks and real industrial datasets, to evaluate its understanding of complex layouts and contexts in documents. The task aims to map each document to structured information consistent with the target ontology or database schema.

Please note that the model card does not provide specific details or scores for each benchmark or dataset. For more detailed information, additional references or documentation may be needed.

#### Factors

The foreseeable characteristics that will influence how the model naver-clova-ix/donut-base-finetuned-docvqa behaves include domain, context, and population subgroups. 

Domain and Context: The model has been evaluated on various Visual Document Understanding (VDU) tasks and datasets, including Document Classification and Document Visual Question Answering (DocVQA). It shows state-of-the-art performance in terms of both speed and accuracy in these domains. It is specifically designed for processing document images and extracting key information from them.

Population Subgroups: The model's behavior may vary across different population subgroups. The references do not provide specific information about the evaluation being disaggregated across factors such as race, gender, or age. Therefore, further analysis is needed to uncover any disparities in performance across different groups.

In summary, the model naver-clova-ix/donut-base-finetuned-docvqa is designed for Visual Document Understanding tasks and has shown competitive performance in domains such as Document Classification and DocVQA. However, further analysis is required to understand its behavior across different population subgroups.

#### Metrics

The metrics used for evaluation in light of tradeoffs between different errors for the model naver-clova-ix/donut-base-finetuned-docvqa are the field-level F1 score and the Tree Edit Distance (TED) based accuracy. 

The field-level F1 score measures whether the extracted field information matches the ground truth. Even if a single character is missed, the score assumes the field extraction is failed. However, it does not take into account partial overlaps and cannot measure the predicted structure, such as groups and nested hierarchy.

The TED-based accuracy metric calculates the accuracy score based on the Tree Edit Distance between the predicted and ground truth trees. It considers the overall structure of the document and can be used for any documents represented as trees. The TED-based accuracy score is calculated as max(0, 1−TED(pr, gt)/TED(ϕ, gt)), where gt, pr, and ϕ stand for ground truth, predicted, and empty trees respectively.

These metrics are used to evaluate the performance of the model in terms of extracting key information and predicting complex structures among field information.

### Results

The evaluation results of the model naver-clova-ix/donut-base-finetuned-docvqa based on the factors and metrics are as follows:

1. Document Classification: The model achieves state-of-the-art performance in document classification, surpassing other general-purpose VDU models such as LayoutLM and LayoutLMv2 in terms of accuracy and speed. It outperforms Lay-outLMv2 accuracy reported in [64] while using fewer parameters and being 2x faster. The performance is comparable to OCR-based models that rely on external OCR engines and is robust even with handwritten documents. [^1^]

2. Document Visual Question Answering (DocVQA): The model achieves competitive scores in the DocVQA dataset. It performs well compared to the general-purpose VDU backbones like LayoutLMv2[^3^]. However, specific metrics or scores are not mentioned in the provided references.

3. Document Information Extraction: The model shows results on four different document IE tasks. However, specific metrics or scores are not mentioned in the provided references[^5^].

In summary, the model naver-clova-ix/donut-base-finetuned-docvqa demonstrates state-of-the-art performance in document classification and competitive scores in Document Visual Question Answering. However, specific metrics or scores for Document Visual Question Answering and Document Information Extraction are not mentioned in the provided references.

#### Summary

The evaluation results for the model naver-clova-ix/donut-base-finetuned-docvqa are as follows:

1. Document Visual Question Answering (DocVQA): Donut achieves competitive scores with baselines that rely on external OCR engines. It shows stability regardless of dataset size and task complexity, making it suitable for industrial use. It is particularly robust in processing handwritten documents.

2. Document Classification: Donut demonstrates state-of-the-art performance among general-purpose VDU models like LayoutLM and LayoutLMv2. It surpasses LayoutLMv2 accuracy while using fewer parameters and running 2x faster. Donut does not rely on external OCR engines.

3. Document Information Extraction (IE): Donut shows the best scores among comparing models in various document IE tasks. It understands complex layouts and contexts in documents, mapping them to structured information consistent with target ontology or database schema.

4. Overall: Donut exhibits strong understanding ability through extensive evaluation on various VDU tasks and datasets. It achieves state-of-the-art performance in terms of both speed and accuracy without relying on OCR engines.

Please note that more detailed information may be available in the referenced sections.

## Model Examination

The model naver-clova-ix/donut-base-finetuned-docvqa is an OCR-free VDU (Visual Document Understanding) model that achieves state-of-the-art performances on various VDU tasks in terms of both speed and accuracy. It demonstrates a strong understanding ability through extensive evaluation on various VDU tasks and datasets.

In terms of explainability/interpretability, there is no specific information provided in the given references about this aspect of the model. Therefore, [More Information Needed].

For more details about the model, including code, trained model, and synthetic data, you can refer to the GitHub repository at https://github.com/clovaai/donut.

## Environmental Impact

- **Hardware Type:** The information about the hardware type used to train the model naver-clova-ix/donut-base-finetuned-docvqa is not provided in the given references. [More Information Needed]
- **Software Type:** The model naver-clova-ix/donut-base-finetuned-docvqa is trained as a visual language model over document images.
- **Hours used:** The amount of time used to train the model naver-clova-ix/donut-base-finetuned-docvqa is not mentioned in the given references. [More Information Needed]
- **Cloud Provider:** The cloud provider that the model naver-clova-ix/donut-base-finetuned-docvqa is trained on is not explicitly mentioned in the provided references. [More Information Needed]
- **Carbon Emitted:** The amount of carbon emitted when training the model naver-clova-ix/donut-base-finetuned-docvqa is not provided in the given references. [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The model architecture of naver-clova-ix/donut-base-finetuned-docvqa is based on Donut, which is an OCR-free Visual Document Understanding (VDU) model. Donut consists of a Transformer-based visual encoder and a textual decoder.

The visual encoder is responsible for extracting features from the input document image. It converts the image into a set of embeddings {z_i | z_i ∈ R^d , 1 ≤ i ≤ n}, where n is the feature map size or the number of image patches, and d is the dimension of the latent vectors of the encoder. The Swin Transformer is used as the visual encoder in this model.

The textual decoder, based on the BART architecture, maps the derived features from the visual encoder into a sequence of subword tokens. The decoder generates a token sequence (y_i) m i=1, where y_i ∈ R^v is a one-hot vector for the i-th token, v is the size of the token vocabulary, and m is a hyperparameter.

The objective of the model is to minimize the cross-entropy loss of the next token prediction by jointly conditioning on the image and previous contexts. This task can be interpreted as a pseudo-OCR task, where the model is trained to read all the texts in the image in reading order (from top-left to bottom-right). The model is trained as a visual language model over visual corpora, i.e., document images.

To summarize:
- Model Architecture: Donut (Transformer-based visual encoder + BART-based textual decoder)
- Objective: Minimize cross-entropy loss of next token prediction by jointly conditioning on the image and previous contexts, trained as a visual language model over document images.

[More Information Needed]

### Compute Infrastructure

The compute infrastructure for the model naver-clova-ix/donut-base-finetuned-docvqa is not explicitly mentioned in the provided references. Therefore, we need more information to determine the exact compute infrastructure used for training and fine-tuning the model.

## Citation

```
@misc{geewook-ocrfree,
    author = {Geewook Kim and
              Hong and
              Moonbin Yim and
              Jeongyeon Nam and
              Jinyoung Park and
              Jinyeong Yim and
              Wonseok Hwang and
              Sangdoo Yun and
              Dongyoon Han and
              Seunghyun Park and
              Naver Clova and
              Naver Search and
              Naver Ai Lab and
              Tmax and
              Google and
              Lbox},
    title  = {OCR-free Document Understanding Transformer},
    url    = {https://arxiv.org/pdf/2111.15664.pdf}
}
```
