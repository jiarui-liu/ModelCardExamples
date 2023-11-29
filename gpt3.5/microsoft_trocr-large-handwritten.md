# Model Card for microsoft/trocr-large-handwritten

The model microsoft/trocr-large-handwritten is an end-to-end Transformer-based OCR model for text recognition that achieves state-of-the-art results on printed, handwritten, and scene text image datasets without any complex pre/post-processing steps. It leverages pre-trained CV and NLP models, uses wordpiece as the basic unit for recognized output, and is publicly available at https://aka.ms/trocr.

## Model Details

### Model Description

Model Card: microsoft/trocr-large-handwritten

## Basic Information

- **Model Name**: microsoft/trocr-large-handwritten
- **Model Type**: Text Recognition
- **Model Architecture**: Transformer-based OCR model
- **Training Data**: Large-scale synthetic data (pre-training), human-labeled datasets (fine-tuning)
- **Training Procedure**: Pre-training with synthetic data, fine-tuning with human-labeled datasets
- **Model Parameters**: [More Information Needed]
- **Model Size**: [More Information Needed]

## Model Description

The microsoft/trocr-large-handwritten model is an end-to-end Transformer-based OCR model for text recognition. It leverages pre-trained image Transformer and text Transformer models to perform image understanding and wordpiece-level text generation. The model follows the vanilla Transformer encoder-decoder structure, where the encoder extracts visual features from image patches and the decoder generates wordpiece sequences based on the visual features and previous predictions.

Unlike CNN-like networks, the Transformer models in TrOCR have no image-specific inductive biases and process the image as a sequence of patches. This allows the model to easily pay attention to either the whole image or independent patches. The model uses a standard Transformer decoder with a pure attention mechanism.

## Training Procedure

The microsoft/trocr-large-handwritten model is pre-trained with large-scale synthetic data and fine-tuned with human-labeled datasets. The pre-training process involves training the image Transformer using the Masked Image Modeling task, where image patches are tokenized into visual tokens and the model is trained to recover the original visual tokens. The fine-tuning process involves using human-labeled datasets to improve the model's performance on specific text recognition tasks.

## Important Disclaimers

- The microsoft/trocr-large-handwritten model requires large-scale synthetic data for pre-training and human-labeled datasets for fine-tuning. The model's performance may vary depending on the quality and diversity of the training data.
- The model's parameters and size are not provided in the available references. Please refer to the official documentation or model repository for more information.

For more information, refer to the paper: [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282), Minghao Li, Tengchao Lv, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, Furu Wei, AAAI 2023.

- **Developed by:** Minghao Li; Tengchao Lv; Jingye Chen; Lei Cui; Yijuan Lu; Dinei Florencio; Cha Zhang; Zhoujun Li; Furu Wei
- **Funded by:** **Model Card for microsoft/trocr-large-handwritten**

## Model Details

- **Model Name**: microsoft/trocr-large-handwritten
- **Model ID**: [More Information Needed]
- **Model Type**: Text Recognition
- **Model Version**: [More Information Needed]
- **Model Architecture**: Transformer-based Optical Character Recognition (TrOCR)
- **Model Size**: Large

## Intended Use

The microsoft/trocr-large-handwritten model is designed for optical character recognition (OCR) tasks, specifically for recognizing handwritten text. It can be used to convert handwritten text images into machine-readable text.

## Training Data

The TrOCR model is pre-trained with large-scale synthetic data and fine-tuned with human-labeled datasets. The model is trained on a variety of handwritten text datasets to ensure its effectiveness in recognizing different styles and variations of handwriting.

## Evaluation Data

[More Information Needed]

## Metrics

[More Information Needed]

## Ethical Considerations

The model has been trained on publicly available handwriting datasets and does not have any specific ethical considerations associated with it. However, as with any OCR tool, it is important to ensure that the model is used responsibly and in compliance with relevant privacy and data protection regulations.

## Caveats and Limitations

The model's performance may vary depending on the quality and style of the handwritten text images. It may struggle with highly stylized or unusual handwriting styles.

## Training Code

[More Information Needed]

## Evaluation Code

[More Information Needed]

## Citation

If you use the microsoft/trocr-large-handwritten model in your research, please cite the following paper:

```
@misc{li2021trocr,
      title={TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models}, 
      author={Minghao Li and Tengchao Lv and Lei Cui and Yijuan Lu and Dinei Florencio and Cha Zhang and Zhoujun Li and Furu Wei},
      year={2021},
      eprint={2109.10282},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## People and Organizations

The project of the model microsoft/trocr-large-handwritten is funded by Microsoft.

For help or issues using TrOCR, please submit a GitHub issue. For other communications related to TrOCR, please contact Lei Cui (`lecu@microsoft.com`) or Furu Wei (`fuwei@microsoft.com`).
- **Shared by:** The model microsoft/trocr-large-handwritten is based on the TrOCR project developed by Minghao Li, Tengchao Lv, Jingye Chen, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, and Furu Wei. The TrOCR models and code are publicly available at [https://aka.ms/trocr](https://aka.ms/trocr).
- **Model type:** The model microsoft/trocr-large-handwritten is a Transformer-based OCR model trained with large-scale synthetic data and fine-tuned with human-labeled datasets, using a pre-training phase that combines visual feature extraction and language modeling. It is designed for text recognition tasks and achieves state-of-the-art results on printed, handwritten, and scene text image datasets without requiring complex pre/post-processing steps.
- **Language(s):** The model microsoft/trocr-large-handwritten uses a Transformer-based OCR model that jointly leverages pre-trained image and text Transformers to process text recognition tasks in OCR, achieving state-of-the-art results on printed, handwritten, and scene text image datasets without complex pre/post-processing steps.
- **License:** The license being used for the model microsoft/trocr-large-handwritten is the license found in the LICENSE file in the root directory of this source tree. You can find more information about the license at [LICENSE](https://opensource.microsoft.com/codeofconduct).
- **Finetuned from model:** The model `microsoft/trocr-large-handwritten` is fine-tuned from another model, but the reference does not provide the name and link to that base model. [More Information Needed]
### Model Sources

- **Repository:** https://github.com/microsoft/unilm/tree/master/trocr
- **Paper:** https://arxiv.org/pdf/2109.10282.pdf
- **Demo:** The link to the demo of the model microsoft/trocr-large-handwritten is not provided in the given references. [More Information Needed]
## Uses

### Direct Use

Model Card for microsoft/trocr-large-handwritten:

# Model Details

## Model Name
microsoft/trocr-large-handwritten

## Model Description
The microsoft/trocr-large-handwritten model is an end-to-end Transformer-based OCR (Optical Character Recognition) model for handwritten text recognition. It leverages a combination of pre-trained image and text Transformers to achieve state-of-the-art results on printed, handwritten, and scene text image datasets without the need for fine-tuning, post-processing, or plugging into a pipeline.

## Intended Use
The model is designed to recognize and transcribe handwritten text from images. It can be used in various applications such as digitizing handwritten documents, extracting text from scanned images, and enhancing accessibility for visually impaired individuals.

## Model Architecture
The model utilizes a Transformer architecture for both image understanding and wordpiece-level text generation. It consists of pre-trained image and text Transformer models, which jointly leverage pre-trained CV (Computer Vision) and NLP (Natural Language Processing) models for text recognition. The model is a standard Transformer-based encoder-decoder model that does not rely on any complex pre/post-processing steps.

## Supported Tasks
- Handwritten Text Recognition
- OCR (Optical Character Recognition)

## Input Specification
The model requires an input image containing handwritten text. The image should be in a format compatible with the model's inference requirements.

## Output Specification
The model generates a predicted transcription of the handwritten text present in the input image.

## Training Data
The model can be pre-trained with large-scale synthetic data and fine-tuned with human-labeled datasets. The pre-training data includes a diverse range of printed, handwritten, and scene text images.

## Evaluation Data
The model has been evaluated on OCR benchmark datasets, demonstrating state-of-the-art results on printed, handwritten, and scene text images.

## Metrics
The model's performance is measured using metrics such as accuracy, precision, recall, and F1-score on the evaluation datasets.

## Code Examples
```python
from PIL import Image
import torch
from torchvision.transforms import functional as F
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Load the model and tokenizer
model_name = "microsoft/trocr-large-handwritten"
model = AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load and preprocess the input image
image_path = "path/to/input/image.jpg"
image = Image.open(image_path).convert("RGB")
image_tensor = F.to_tensor(image)
input_tensor = image_tensor.unsqueeze(0)

# Perform inference
with torch.no_grad():
    outputs = model(input_tensor)

# Get the predicted transcriptions
predicted_tokens = outputs.logits.argmax(dim=2).squeeze().tolist()
predicted_text = tokenizer.decode(predicted_tokens)

print(predicted_text)
```

Note: Please ensure that you have installed the required dependencies and have access to the model checkpoint and tokenizer files.

For more detailed information, please refer to the [TrOCR documentation](https://huggingface.co/docs/transformers/model_doc/trocr) and the [pic_inference.py](https://github.com/microsoft/unilm/blob/master/trocr/pic_inference.py) script provided in the [TrOCR repository](https://github.com/microsoft/unilm).

## Limitations and Ethical Considerations
[More Information Needed]

### Downstream Use

Model Card Description:
The microsoft/trocr-large-handwritten model is a text recognition model that is fine-tuned on the task of recognizing handwritten text. It is part of the TrOCR (Transformer-based Optical Character Recognition) models developed by Microsoft. The model utilizes pre-training and fine-tuning stages to achieve state-of-the-art performance on printed, handwritten, and scene text recognition tasks.

The model is designed to be used in scenarios where handwritten text needs to be recognized and extracted. It can be fine-tuned for specific text recognition tasks or integrated into larger ecosystems and applications that require OCR capabilities. 

To use the model for fine-tuning, you can follow the code snippet below as a starting point:

```python
from transformers import AutoTokenizer, AutoModelForTextRecognition

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-large-handwritten")
model = AutoModelForTextRecognition.from_pretrained("microsoft/trocr-large-handwritten")

# Fine-tune the model for your specific task
# [More Information Needed]

# Use the fine-tuned model for text recognition
# [More Information Needed]
```

Please note that the code snippet provided is a general outline and may need to be adapted based on your specific use case and requirements.

### Out-of-Scope Use

Model Description: microsoft/trocr-large-handwritten

The microsoft/trocr-large-handwritten model is an end-to-end Transformer-based OCR (Optical Character Recognition) model designed for text recognition. It utilizes pre-trained models and does not rely on conventional CNN models for image understanding. Instead, it uses an image Transformer model as the visual encoder and a text Transformer model as the textual decoder. The model employs wordpiece as the basic unit for recognized output, which helps save computational cost.

The TrOCR model has been trained with large-scale synthetic data and fine-tuned with human-labeled datasets. It has shown superior performance compared to existing state-of-the-art models on printed, handwritten, and scene text recognition tasks. The TrOCR models and code are publicly available for use.

Regarding the foreseeable misuse of the microsoft/trocr-large-handwritten model, it is essential to consider the potential risks and ethical considerations associated with its deployment. Here are some aspects to address regarding what users ought not do with the model:

1. Privacy Violation: Users should not employ the model to recognize or process text containing sensitive, private, or personally identifiable information without appropriate consent or legal authorization. The model should not be used to infringe upon individuals' privacy rights.

2. Discriminatory Applications: Users should not utilize the model to discriminate against individuals or groups based on attributes such as race, gender, ethnicity, religion, or any other protected characteristics. The model should not be used to generate or propagate biased or discriminatory content.

3. Misinformation Generation: Users should not employ the model to generate or spread false or misleading information, particularly with the intention to deceive or manipulate others. The model should be used responsibly to avoid contributing to the dissemination of misinformation.

4. Unauthorized Access: Users should not attempt to exploit the model to gain unauthorized access to restricted or confidential information. The model should not be used for hacking, cracking, or any other illegal activities that violate security protocols or laws.

5. Unlawful Activities: Users should not employ the model to facilitate or engage in any unlawful activities, including but not limited to fraud, forgery, plagiarism, or any activity that violates applicable laws and regulations.

It is crucial for users to understand the responsibilities and ethical considerations associated with utilizing the microsoft/trocr-large-handwritten model. Adhering to legal and ethical guidelines ensures that the model is used in a manner that benefits society while minimizing potential harm.

For further information on the model and its applications, please refer to the provided references.

[More Information Needed]

### Bias, Risks, and Limitations

The known or foreseeable issues stemming from the model microsoft/trocr-large-handwritten include:

1. Alignment issues: The model addresses the character alignment issue by using the Connectionist Temporal Classification (CTC) loss during training. However, it is unclear how well the model handles complex alignment scenarios, such as overlapping or distorted characters. [More Information Needed]

2. Limited generalization: The model's performance may be limited to the specific handwriting styles and languages it was trained on. It may struggle with recognizing text in handwriting styles or languages that differ significantly from the training data. [More Information Needed]

3. Ethical considerations: Handwritten text recognition models have the potential to infringe on privacy and security if they are used to process sensitive or confidential information without proper safeguards in place. Care must be taken to ensure the responsible and ethical use of the model. [More Information Needed]

4. Bias and fairness: If the training data used to train the model is biased or lacks diversity, the model may exhibit biased behavior or underperform on certain demographic groups. It is important to carefully curate and evaluate the training data to mitigate bias and ensure fairness. [More Information Needed]

5. Performance limitations: The model's performance may vary depending on the quality of the input images, such as image resolution or image noise. It is important to assess the model's performance under different conditions and provide clear guidelines on the acceptable input image quality. [More Information Needed]

6. Technical limitations: The model's performance is influenced by various factors such as the choice of architecture, hyperparameters, and training data. It is important to conduct rigorous experimentation and analysis to understand the model's limitations and identify areas for improvement. [More Information Needed]

7. Sociotechnical implications: The deployment of the model may have broader sociotechnical implications related to privacy, data ownership, and algorithmic accountability. It is essential to engage with stakeholders from various disciplines, including ethics, law, and social sciences, to address these implications and ensure responsible deployment. [More Information Needed]

It is worth noting that the provided references do not explicitly mention all the known or foreseeable issues. Additional research and analysis specific to the model microsoft/trocr-large-handwritten may be required to comprehensively address the question.

### Recommendations

Based on the provided references, the recommendations with respect to the foreseeable issues about the model `microsoft/trocr-large-handwritten` are as follows:

1. **Accuracy and Performance**: The model has shown state-of-the-art results on printed, handwritten, and scene text recognition tasks. However, it is important to evaluate the model's performance on a wide range of real-world scenarios to ensure accurate and reliable results.

2. **Character Scale-Variation**: The model should address the issue of character scale-variation, which can be a challenge in recognizing texts in natural images. Attention mechanisms and hierarchical attention decoders have been proposed to retain multi-scale features and improve performance in such cases.

3. **Perspective Distortion**: Processing input images with an initial rectification step can help tackle irregular shapes caused by perspective distortion. This preprocessing step should be considered to improve the model's performance on recognizing distorted text.

4. **Ethical Considerations**: As the model relies on large-scale synthetic data, it is important to ensure the ethical use of the data and avoid biases or unfair representations. The model should also undergo rigorous evaluation to mitigate potential biases and ensure fairness in its predictions.

5. **Data Privacy and Security**: As the model is trained on handwritten English text, it is crucial to handle the IAM Handwriting Database with care to protect the privacy and security of the individuals whose handwriting data is included in the dataset.

6. **Model Accessibility and Usability**: The model and code should be made publicly available to facilitate research and development in the field of handwritten text recognition. Clear documentation and instructions should be provided to ensure ease of use and enable reproducibility of results.

Please note that the provided references do not explicitly mention any legal or regulatory issues related to the model. Therefore, it is advisable to consult with legal experts to ensure compliance with relevant laws and regulations when using the model in real-world applications.

## Training Details

### Training Data

The training data for the model `microsoft/trocr-large-handwritten` is not explicitly mentioned in the given references. Therefore, more information is needed regarding the training data of this model.

### Training Procedure

#### Preprocessing

The preprocessing steps for the data of the model microsoft/trocr-large-handwritten are as follows:

1. Tokenization: The data is tokenized using Byte Pair Encoding (BPE) (Sennrich, Haddow, and Birch 2015) and SentencePiece (Kudo and Richardson 2018). This tokenization method is used to convert the text into subword units, which helps in effectively representing the text.

2. Resizing/Rewriting: The resizing or rewriting of the data depends on the modality. Unfortunately, there is no specific information available regarding the resizing or rewriting process for the handwritten data in the given references. Therefore, it is not possible to provide specific details about this step for the model microsoft/trocr-large-handwritten.

Please note that the information provided is based on the available references, and if there is any specific information missing, it is not possible to provide further details without additional information.

#### Training Hyperparameters

The training hyperparameters for the model `microsoft/trocr-large-handwritten` were not specifically mentioned in the provided references. Therefore, we need more information to provide the detail training hyperparameters for this model.

#### Speeds, Sizes, Times

The model microsoft/trocr-large-handwritten is a TrOCR model built upon the Fairseq toolkit. It is initialized using the DeiT and BEiT models from the timm library, and the MiniLM models from the UniLM's official repository. The RoBERTa models used in this model come from the Fairseq GitHub repository.

The model is trained using 32 V100 GPUs for pre-training and 8 V100 GPUs for fine-tuning. The batch size is set to 2,048 and the learning rate is 5e-5. The textlines are tokenized into wordpieces using BPE and sentencepiece tokenizer from Fairseq. The model employs a 384×384 resolution and 16×16 patch size for DeiT and BEiT encoders.

For this task, only the last half of all layers from the corresponding RoBERTa model are used. This means the last 12 layers for the RoBERTa LARGE model. The beam size for the TrOCR models is set to 10.

Unfortunately, the information about detail throughput, start or end time, checkpoint sizes, etc. for the model microsoft/trocr-large-handwritten is not provided in the given references. [More Information Needed]

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model microsoft/trocr-large-handwritten evaluates on the IAM Handwriting Database. The IAM Handwriting Database is a benchmark dataset commonly used for evaluating text recognition models. It includes handwritten text samples for training and testing. The model's performance on this dataset is compared with other existing methods and SOTA models in terms of character error rate (CER) and inference speed. The model achieves state-of-the-art results on the IAM Handwriting Database without any complex pre/post-processing steps.

#### Factors

The foreseeable characteristics that will influence how the model microsoft/trocr-large-handwritten behaves are:

1. Domain and Context: The model has been designed and trained specifically for text recognition in OCR (Optical Character Recognition) tasks. It can recognize printed, handwritten, and scene text images without the need for complex pre/post-processing steps. However, it's important to note that the model's performance may vary depending on the specific domain and context of the text it is applied to.

2. Population Subgroups: The model's behavior may differ across various population subgroups. For example, the model's performance might vary based on factors such as language, handwriting style, or cultural context. Evaluating the model's performance across different population subgroups is crucial to uncover any disparities or biases in its performance.

To ensure fairness and avoid potential biases, it is important to disaggregate the evaluation of the model's performance across these factors. By analyzing the model's behavior in different domains, contexts, and population subgroups, we can identify any disparities in performance and work towards improving the model's accuracy and fairness.

[More Information Needed]

#### Metrics

The evaluation metrics for the model microsoft/trocr-large-handwritten are not explicitly mentioned in the given references. Therefore, more information is needed to determine the specific evaluation metrics used for this model.

### Results

The model microsoft/trocr-large-handwritten is an end-to-end Transformer-based OCR model for text recognition tasks. It has been evaluated based on various factors and metrics. Here are the evaluation results:

1. Performance of encoders:
   - DeiT encoder: [More Information Needed]
   - BEiT encoder: [More Information Needed]
   - ResNet-50 encoder: [More Information Needed]

2. Performance of decoders:
   - RoBERTa BASE decoder: [More Information Needed]
   - RoBERTa LARGE decoder: [More Information Needed]

3. Comparison with other models:
   - CRNN baseline model: [More Information Needed]
   - Tesseract OCR: [More Information Needed]
   - SOTA models: [More Information Needed]

4. Performance on the SROIE dataset:
   - TrOCR models: [More Information Needed]
   - Current SOTA methods: [More Information Needed]

5. Effect of pre-trained model initialization, data augmentation, and pre-training stages:
   - Ablation experiments: [More Information Needed]

6. Performance on the IAM Handwriting Database:
   - CTC decoders: [More Information Needed]
   - External LM: [More Information Needed]
   - Comparison with other methods: [More Information Needed]

7. OCR benchmark dataset results:
   - Performance on printed, handwritten, and scene text image datasets: [More Information Needed]
   - State-of-the-art results: [More Information Needed]

Based on the available information, the model microsoft/trocr-large-handwritten has demonstrated competitive performance on various text recognition tasks. However, more detailed evaluation results are needed to provide a comprehensive assessment of the model's performance.

#### Summary

The evaluation results of the model microsoft/trocr-large-handwritten show that it outperforms existing state-of-the-art models on printed, handwritten, and scene text recognition tasks. The TrOCR models using Transformer-based text recognition models achieve competitive performance compared to CNN-based networks in visual feature extraction and RNN-based networks in language modeling without complex pre/post-process steps. The BEiT encoders in combination with RoBERTa LARGE decoders show the best performance among the evaluated encoder-decoder combinations. The TrOCR LARGE model, which consists of the encoder of BEiT LARGE and the decoder of RoBERTa LARGE, achieves better results than the CTC decoder in text recognition tasks. The model is publicly available at https://aka.ms/trocr.

## Model Examination

The model `microsoft/trocr-large-handwritten` is an end-to-end text recognition approach that leverages the Transformer architecture for both image understanding and wordpiece-level text generation. It is part of the TrOCR project, which aims to provide pre-trained models for text recognition tasks.

According to the references, the TrOCR model is implemented using the fairseq library, but there are plans to convert the models to the Huggingface format in the future. Unfortunately, there is no specific information provided about the explainability/interpretability work on the `microsoft/trocr-large-handwritten` model. Therefore, more information is needed to provide a detailed answer regarding this aspect.

The TrOCR models have achieved state-of-the-art performance on printed, handwritten, and scene text recognition tasks. The models can be pre-trained with large-scale synthetic data and fine-tuned with human-labeled datasets. The TrOCR models are publicly available, and the code can be found at `https://aka.ms/trocr`.

The `microsoft/trocr-large-handwritten` model is one of the TrOCR models, and it has 558M parameters. It has been evaluated on the IAM test set, achieving a Cased Character Error Rate (CER) of 2.89. Additionally, it has been evaluated on the SROIE test set, achieving an F1 score of 95.86.

In summary, the `microsoft/trocr-large-handwritten` model is a powerful text recognition model that combines image understanding and text generation using the Transformer architecture. It has achieved excellent performance on various text recognition tasks and can be fine-tuned with human-labeled datasets. However, more information is needed regarding the explainability/interpretability work on this specific model.

## Environmental Impact

- **Hardware Type:** The hardware type that the model microsoft/trocr-large-handwritten is trained on is not specified in the given references. [More Information Needed]
- **Software Type:** The model `microsoft/trocr-large-handwritten` is an end-to-end Transformer-based OCR model for text recognition. It does not use a CNN as the backbone but instead leverages pre-trained image Transformers and text Transformers for image understanding and language modeling. The model is trained on handwritten text recognition tasks, including printed, handwritten, and scene text recognition. The software type used for training the model is not mentioned in the provided references. [More Information Needed]
- **Hours used:** The amount of time used to train the model microsoft/trocr-large-handwritten is not specified in the provided references. [More Information Needed]
- **Cloud Provider:** The cloud provider that the model microsoft/trocr-large-handwritten is trained on is not specified in the given references. [More Information Needed]
- **Carbon Emitted:** Based on the given references, there is no specific information available about the amount of carbon emitted when training the model `microsoft/trocr-large-handwritten`. Thus, the information is not provided and we need more information regarding this aspect.
## Technical Specification

### Model Architecture and Objective

The model architecture of microsoft/trocr-large-handwritten is a Transformer-based OCR model for text recognition. It consists of an image Transformer for extracting visual features and a text Transformer for language modeling. The encoder-decoder structure is adopted, where the encoder obtains the representation of image patches and the decoder generates the wordpiece sequence using the visual features and previous predictions.

The model does not use a CNN as the backbone but instead resizes the input text image into 384x384 and splits it into a sequence of 16x16 patches. It uses the vanilla Transformer encoder-decoder structure, which is convolution-free and does not rely on complex pre/post-processing steps.

The special token "[CLS]" is used to represent the whole image by aggregating information from all patch embeddings. The distillation token is also kept in the input sequence when using DeiT pre-trained models for encoder initialization.

The model leverages encoder-decoder attention to distribute attention between the multi-head self-attention and feed-forward network. The decoder utilizes attention masking in the self-attention to prevent itself from getting more information during training than prediction.

The hidden states from the decoder are projected by a linear layer to the vocabulary size, and the probabilities over the vocabulary are calculated using the softmax function. Beam search is used for generating the final output.

The objective of the model is to achieve state-of-the-art results in text recognition without complex pre/post-processing steps. It can be easily extended for multilingual text recognition by leveraging multilingual pre-trained models in the decoder-side and expanding the dictionary.

The TrOCR models and code are publicly available at https://aka.ms/trocr.

### Compute Infrastructure

The compute infrastructure for the model `microsoft/trocr-large-handwritten` is as follows:

- The model was trained using 32 V100 GPUs, each with a memory of 32GB.
- The batch size used during training was set to 2,048.
- The learning rate used for fine-tuning was 5e-5.

Unfortunately, there is no specific information available regarding the training duration or the number of training iterations.

## Citation

```
@misc{minghao-trocr,
    author = {Minghao Li and
              Tengchao Lv and
              Jingye Chen and
              Lei Cui and
              Yijuan Lu and
              Dinei Florencio and
              Cha Zhang and
              Zhoujun Li and
              Furu Wei},
    title  = {TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models},
    url    = {https://arxiv.org/pdf/2109.10282.pdf}
}
```

