# Model Card for microsoft/git-base

The model microsoft/git-base is a Generative Image-to-text Transformer (GIT) that simplifies the architecture by using only one image encoder and one text decoder. It is pre-trained on 0.8 billion image-text pairs with the language modeling task and achieves state-of-the-art performance on various challenging benchmarks.

## Model Details

### Model Description

Model Name: microsoft/git-base

## Model Architecture
The microsoft/git-base model is based on the GIT (Generative Image-to-text Transformer) architecture, as described in the paper "GIT: A Generative Image-to-text Transformer for Vision and Language" (reference 1). 

The model architecture consists of an image encoder and a text decoder. The image encoder encodes the input image into a feature representation, while the text decoder generates a textual description based on the encoded image representation.

## Training Procedures
The training procedure for the microsoft/git-base model involves pre-training and fine-tuning.

1. Pre-training: The model is pre-trained using a large dataset and a language modeling objective (reference 9). The pre-training data format is the same as that in the image captioning task, and the same language modeling task is used for fine-tuning (reference 3).

2. Fine-tuning: The model is fine-tuned on specific tasks, such as image classification and image captioning. For image classification, the model is fine-tuned to predict class names as image captions in an auto-regressive way, which allows for handling new data and categories (reference 2). The fine-tuning process involves updating all parameters of the model, including the image encoder and text decoder (reference 7).

## Parameters
Specific details about the parameters of the microsoft/git-base model are not provided in the references. [More Information Needed]

## Important Disclaimers
1. The microsoft/git-base model is not specifically designed for the video domain but can achieve competitive performance with a simple architecture change, which involves encoding multiple frames independently and concatenating the features (reference 5).

2. The generative approach chosen for the model, although beneficial for free-form answer generation, may result in slightly lower performance compared to discriminative approaches in certain tasks, such as VQAv2 (reference 6).

3. The model may contain trademarks or logos that are subject to Microsoft's trademark and brand guidelines (reference 8).

4. The performance of the model, as summarized in Table 1 of the paper, shows new state-of-the-art results across various challenging benchmarks (reference 9).

5. Details about the specific parameters used in the pre-training and fine-tuning processes, such as learning rate schedules and hyperparameters, are not provided. [More Information Needed]

- **Developed by:** Jianfeng Wang; Zhengyuan Yang; Xiaowei Hu; Linjie Li; Kevin Lin; Zicheng Liu; Ce Liu; Lijuan Wang
- **Funded by:** The model card for microsoft/git-base:

# Model Card

## Model Details

- Model Name: microsoft/git-base
- Model Version: [insert version number]
- Model Type: Deep Learning
- Model Architecture: Generative Image-to-text Transformer (GIT)
- Model Purpose: Vision and Language tasks
- Model Author: Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li, Kevin Lin, Zhe Gan, Zicheng Liu, Ce Liu, Lijuan Wang
- Contact Person: [insert contact person name]
- License: [insert license]

## Intended Use

The microsoft/git-base model is designed to generate image captions for vision and language tasks. It can be used for tasks such as image captioning, visual question answering, and image-text retrieval.

## Training Data

The model was trained on a large dataset of image-caption pairs. The specific details of the training data are not provided.

## Evaluation Data

The microsoft/git-base model was evaluated on various benchmarks, achieving state-of-the-art performance across numerous challenging tasks. Please refer to the original paper for detailed evaluation results.

## Ethical Considerations

The microsoft/git-base model has been developed with the objective of improving performance on vision and language tasks. It is important to consider potential biases and limitations of the model when using it in real-world applications. Further analysis and evaluation are recommended to assess the model's performance in specific use cases.

## Caveats and Known Limitations

The limitations of the microsoft/git-base model are not specified in the provided references. Further analysis and evaluation are recommended to understand the model's performance in various scenarios.

## Dependencies

The model code is based on various open-source projects, including transformers, CLIP, maskrcnn-benchmark, Oscar, and virtex. Please refer to the respective repositories for more information.

## Funding

The funding details for the microsoft/git-base project are not specified in the provided references. [More Information Needed]

## Citation

If you find the microsoft/git-base model helpful, please consider citing the following reference:

```
@article{wang2022git,
  title={GIT: A Generative Image-to-text Transformer for Vision and Language},
  author={Wang, Jianfeng and Yang, Zhengyuan and Hu, Xiaowei and Li, Linjie and Lin, Kevin and Gan, Zhe and Liu, Zicheng and Liu, Ce and Wang, Lijuan},
  journal={arXiv preprint arXiv:2205.14100},
  year={2022}
}
```

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft's Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos is subject to those third-party's policies.

Please note that this model card is subject to updates and changes. For the most up-to-date information, please refer to the provided references and contact the designated contact person.
- **Shared by:** The contributors who made the model microsoft/git-base available online as a GitHub repo are not mentioned in the given references. [More Information Needed]
- **Model type:** The model microsoft/git-base is a generative image-to-text transformer (GIT) that is trained using a language modeling task on 0.8 billion image-text pairs, with an image encoder based on a contrastive pre-trained model and a text decoder following the architecture of GPT3.
- **Language(s):** The model microsoft/git-base processes natural human language in the form of image captions and associated text descriptions.
- **License:** The name and link to the license being used for the model microsoft/git-base is the Apache License 2.0. You can find more information about the license [here](https://opensource.org/licenses/Apache-2.0).
- **Finetuned from model:** Based on the provided information, it is not mentioned which base model was used to fine-tune the microsoft/git-base model. Therefore, the answer to the question about the base model for microsoft/git-base is "[More Information Needed]".
### Model Sources

- **Repository:** https://github.com/microsoft/GenerativeImage2Text
- **Paper:** https://arxiv.org/pdf/2205.14100.pdf
- **Demo:** The link to the demo of the model microsoft/git-base is not provided in the given references. [More Information Needed]
## Uses

### Direct Use

The model microsoft/git-base can be used without fine-tuning, post-processing, or plugging into a pipeline by utilizing its decoder-only language model capability, similar to GPT3. This allows the model to generate text based on input prompts without the need for additional processing.

To use the model, you can follow these steps:

1. Install the necessary dependencies by running the following commands:
```shell
pip install -r requirements.txt
python setup.py build develop
```

2. Once the dependencies are installed, you can use the model for text generation by providing a prompt. Here is an example code snippet:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "microsoft/git-base"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

prompt = "Enter your prompt here"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

output = model.generate(input_ids, max_length=100)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

Please note that this code snippet assumes you have installed the necessary dependencies and have the `GPT2LMHeadModel` and `GPT2Tokenizer` classes from the Hugging Face `transformers` library. If you encounter any issues, please refer to the model's documentation or seek further assistance.

[More Information Needed]

### Downstream Use

The microsoft/git-base model can be fine-tuned for various tasks or plugged into a larger ecosystem or app. When fine-tuned, it can be used for tasks such as image captioning and question answering. 

For image captioning, the model can be fine-tuned on the COCO dataset using the same language modeling (LM) task as pre-training. The training data format remains the same as in pre-training. The fine-tuned model, called GIT_BASE_COCO, achieves a CIDEr score of 131.4. Here is an example code snippet for fine-tuning the model on COCO:

```python
python -m generativeimage2text.train -p "{'type': 'forward_backward_example', 'image_files': ['aux_data/images/1.jpg', 'aux_data/images/2.jpg'], 'dataset': 'COCO'}"
```

For question answering, the model can be fine-tuned on the MSRVTT dataset. The fine-tuned model, called GIT_BASE_MSRVTT_QA, achieves an accuracy of 41.0 on the MSRVTT dataset. Unfortunately, there is no code snippet provided for fine-tuning on MSRVTT.

To use the model in a larger ecosystem or app, the code provided in the repository can be easily plugged into any trainer. The repository contains the key code path for constructing the network input with transformations and forward/backward operations. Here is an example code snippet for using the base model:

```python
# Install azfuse if not already installed
pip install git+https://github.com/microsoft/azfuse.git

# Clone the GenerativeImage2Text repository
git clone https://github.com/microsoft/GenerativeImage2Text.git
cd GenerativeImage2Text

# Run the training script
python -m generativeimage2text.train -p "{'type': 'forward_backward_example', 'image_files': ['aux_data/images/1.jpg', 'aux_data/images/2.jpg']}"
```

Please note that the above code snippet is for the base model and not specifically for the fine-tuned versions mentioned earlier. For using the fine-tuned models, additional code specific to the respective fine-tuning task would be required.

In summary, the microsoft/git-base model can be fine-tuned for tasks like image captioning and question answering, and it can be easily integrated into a larger ecosystem or app by using the provided code snippets.

### Out-of-Scope Use

The model microsoft/git-base, which has been pre-trained on large-scale data, is designed to improve the performance and assist visually-impaired individuals. However, there are certain considerations regarding its potential misuse that need to be addressed:

1. Toxic Language: The data used for pre-training the model are not guaranteed to be free from toxic language. This raises concerns about the potential of the model to generate outputs that may be harmful or offensive. Careful attention should be given to deploying the model in real-world applications, and additional research is required to mitigate the risk of toxic language in the model's output.

2. Legal and Ethical Use: Users of the microsoft/git-base model should be mindful of legal and ethical regulations when utilizing the model. The model should be used responsibly and in compliance with applicable laws, ensuring that the generated content does not violate any legal or ethical standards.

3. User Guidelines: It is important to provide clear guidelines and instructions to users of the model on how to use it appropriately. This includes educating users about the potential risks associated with the model's output and providing recommendations on how to interpret and filter the generated content to ensure its suitability for specific applications.

4. Responsible Deployment: Prior to deploying the microsoft/git-base model in production, it is crucial to conduct thorough testing and validation to ensure that the model performs as intended and meets the necessary standards of accuracy, fairness, and safety. Additionally, ongoing monitoring and maintenance should be employed to address any emerging issues and maintain the model's reliability.

In summary, while the microsoft/git-base model offers improvements in assisting visually-impaired individuals, there are potential risks associated with its misuse. It is essential to take precautions, such as deploying the model responsibly, providing user guidelines, and ensuring compliance with legal and ethical standards, to mitigate these risks and promote the responsible use of the model.

### Bias, Risks, and Limitations

Based on the provided references, here are the known or foreseeable issues stemming from the model microsoft/git-base:

1. The model is pre-trained on a large-scale dataset that may contain toxic language, which could potentially poison the output. Although few instances of toxic language have been observed qualitatively, deploying the model in practice requires special care to control the output and further research exploration is needed.

2. The model focuses on improving absolute performance through the pretraining-and-finetuning strategy, but it is unclear how to control the generated captions and perform in-context learning without parameter updates. This limitation is left as future work.

3. The model's performance depends on the scale of the pretraining set. Smaller pretraining datasets may result in lower performance, while larger datasets may require longer training iterations. Effectively training the model from scratch is an area that requires further investigation.

4. The model's strong capability of recognizing and describing various objects and scenes indicates that it has encoded rich multi-modal knowledge about the visual world. However, there may still be limitations and misunderstandings in its understanding and generation of text descriptions.

In summary, the known or foreseeable issues with microsoft/git-base include potential toxicity in the output, limitations in controlling generated captions, dependency on pretraining dataset scale, and the possibility of limitations and misunderstandings in text description understanding and generation. Further research and care are necessary to address these issues.

### Recommendations

Based on the provided references, here are the recommendations with respect to the foreseeable issues about the model microsoft/git-base:

1. Special care should be taken when deploying the model in practice due to the potential presence of toxic language in the data used for pre-training. More research exploration is required to control the output and ensure it does not produce harmful or offensive content.

2. The base model's performance may drop when trained on a large amount of data (0.8B) compared to a smaller amount (14M) that is more similar to the target dataset (COCO). This indicates that the model's effectiveness may vary depending on the scale and similarity of the training data. Further investigation is needed to understand and mitigate this issue.

3. As the model achieves state-of-the-art performance on image captioning and question answering tasks, it is important to consider the potential impact of surpassing human performance. Ethical considerations must be taken into account to ensure responsible deployment and minimize unintended consequences.

4. The model's strong capability of recognizing and describing various visual elements, such as text, tables, charts, food, banknotes, logos, landmarks, characters, celebrities, and products, suggests that it has encoded rich multi-modal knowledge about the visual world. However, it is important to validate and address any biases or limitations in this encoding to avoid perpetuating stereotypes or discriminatory behaviors.

5. Longer training iterations and a larger pretraining dataset may be necessary to effectively train the model from scratch. Future work should focus on understanding the impact of these factors and finding ways to improve the model's performance in such scenarios.

In summary, it is recommended to carefully monitor and control the model's output, consider the impact of surpassing human performance, address potential biases or limitations, and continue research efforts to improve the model's training process.

## Training Details

### Training Data

The training data of the model microsoft/git-base consists of pre-training samples that contain scene text descriptions. The data is preprocessed to ensure that the shorter length of the image is no larger than 384 and the longer side is no larger than 640 while maintaining the aspect ratio. The images are also re-saved in JPEG format with a quality of 90. For more details on the data preprocessing, please refer to [Reference 2](More Information Needed).

### Training Procedure

#### Preprocessing

The preprocessing steps for the data of the model microsoft/git-base involve tokenization, resizing, and rewriting, depending on the modality. Here are the details:

1. Tokenization:
The text data is tokenized using a language modeling (LM) loss. The loss is computed using cross-entropy (CE) with label smoothing of 0.1. The tokens include the image captions as class names and the special tokens [BOS] (beginning of sequence) and [EOS] (end of sequence).

2. Resizing/Rewriting:
For image-text pairs, the images are resized based on the longer edge to a size of 384. The images are also padded to form a square shape.

3. OCR and Scene Text Recognition:
The model does not rely on an OCR engine or dynamic pointer network for scene text recognition. Instead, it learns to read the scene text through large-scale pre-training. This approach achieves new state-of-the-art (SoTA) performance on scene-text-related visual question answering (VQA) tasks.

4. Decoder-Only Language Model:
Without the image input, the model functions as a decoder-only language model, similar to GPT3. This design allows for the possibility of leveraging text-only data to enhance decoding capabilities with a scaled-up decoder, although this is left as future work.

5. Initialization:
The image encoder is initialized from contrastive pretraining. The image encoder is based on the base-sized version of the Vision Transformer (ViT) and is initialized from the CLIP model, supervised pretraining (classification task on ImageNet), self-supervised pretraining (MAE on ImageNet), or randomly initialized.

Please note that there may be additional details not covered in the provided references. It's always good to refer to the official documentation or papers for more comprehensive information on the model's preprocessing steps.

#### Training Hyperparameters

The training hyperparameters for the model `microsoft/git-base` are not explicitly mentioned in the provided references. Therefore, more information is needed to answer the question.

#### Speeds, Sizes, Times

The details about the model microsoft/git-base are not provided in the given references. [More Information Needed]

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model microsoft/git-base evaluates on the following benchmarks or datasets:

1. ImageNet-1k: The model is fine-tuned on ImageNet-1k, where each category is mapped to a unique class name. The prediction is considered correct only if it exactly matches the ground-truth label, subject to more or fewer whitespaces.

2. Zero-shot/Few-shot: The model's performance is evaluated on a task where it has no prior knowledge of the vocabulary. The model's exactly-match accuracy is low at 1.93%. However, if the requirement is relaxed and the prediction contains the ground-truth, the accuracy increases to 40.88%.

3. VQA (Visual Question Answering): The model's performance is compared with other models on various VQA evaluation benchmarks, including VQAv2, TextVQA, VizWiz-VQA, ST-VQA, and OCR-VQA. The model achieves higher accuracy compared to Flamingo.

4. Captioning and QA: The model's performance is evaluated on captioning and QA tasks. Although the model's performance is not explicitly mentioned, it is indicated to be better than VL (Vision and Language) pretraining and Flamingo with 80B parameters.

5. Scene Text Recognition: The model's performance is evaluated on scene text recognition, where it can recognize scene text and describe it in natural language form.

Please note that further details or specific metrics for each benchmark or dataset are not provided in the given references.

#### Factors

The foreseeable characteristics that will influence how the model microsoft/git-base behaves include:

1. Domain and Context: The model has been trained on 0.8 billion image-text pairs using the language modeling task. It has demonstrated impressive performance in recognizing and describing various visual elements such as scene text, tables, charts, food, banknotes, logos, landmarks, characters, celebrities, products, etc. This indicates that the model has encoded rich multi-modal knowledge about the visual world.

2. Population Subgroups: The model's performance should ideally be evaluated and disaggregated across different factors to uncover any disparities in performance. For example, the model's performance can be evaluated separately on images annotated with male and female subjects. The normalized performance difference (NPD) can be calculated to determine if the model shows bias towards one group over the other. The evaluation results indicate that the bias ranges only from 0.7% to 5.3% across all metrics, suggesting a relatively balanced performance across gender.

3. Scalability: The performance of the model is significantly improved by scaling up the pre-training data and the model size. However, it is mentioned that with smaller-scale pre-training sets, the model's performance may be lower. It is suggested that a larger dataset may reduce this performance gap, but it may require longer training iterations. Further research is needed to explore effective training strategies for the model from scratch.

In summary, the model microsoft/git-base is expected to perform well in recognizing and describing various visual elements. Evaluation should be carried out to identify any biases or disparities in performance across population subgroups. Additionally, the model's scalability and training strategies need to be further investigated.

#### Metrics

The metrics used for evaluation in light of tradeoffs between different errors for the model microsoft/git-base are not explicitly mentioned in the provided references. [More Information Needed]

### Results

Based on the provided references, there is no specific information about the evaluation results of the model microsoft/git-base. Therefore, the evaluation results for the model are not available. [More Information Needed]

#### Summary

The evaluation results for the model microsoft/git-base are as follows:

1. On the ImageNet-1k dataset, the model achieved descent accuracy without pre-defining the vocabulary. However, when compared to the Florence model, it performed slightly worse, possibly due to the generative approach requiring prediction of more tokens. [Reference 1]

2. In zero-shot/few-shot scenarios, where the model had no knowledge of the vocabulary, the exactly-match accuracy was only 1.93%. However, if the requirement was relaxed and the prediction contained the ground-truth, the accuracy increased to 40.88%. This indicates that the model can identify the image content well if the vocabulary is known. [Reference 2]

3. Performance on video captioning tasks was evaluated on various datasets such as MSVD, MSRVTT, YouCook2, VATEX, and TVC. Detailed results can be found in the supplementary materials. [Reference 3]

4. The model outperformed VL pretraining on captioning and QA tasks, achieving better results than Tang et al. (2021) and Flamingo (Alayrac et al., 2022) in terms of captioning performance. [Reference 4]

5. The model achieved state-of-the-art performance on challenging benchmarks, surpassing the discriminative counterpart by around 1 point. The use of RoBerta as the text encoder in the Florence model might have contributed to its improved performance. [Reference 5]

Please note that additional information may be needed to provide a comprehensive summary of the evaluation results for the microsoft/git-base model.

## Model Examination

The model card description for the microsoft/git-base model can be written as follows:

---

# Model Card: microsoft/git-base

## Description
The microsoft/git-base model is a deep learning model developed based on the CLIP model by Radford et al. (2021). It serves as the base and large variants of the CLIP model. The model has been trained and evaluated on various benchmarks such as COCO, TextCaps, and VizWiz-QA.

## Performance
The model performance on COCO dataset varies depending on the amount of training data. While the base model benefits from 4M to 14M data, its performance drops with 0.8B data due to the noise in the majority of the data. Additionally, the base model's limited capacity may not effectively benefit from large-scale data. Similar observations have also been reported for ImageNet-1k classification.

## Bias Analysis
The bias of the microsoft/git-base model has been investigated by calculating the normalized performance difference (NPD). NPD measures the difference in performance metrics (e.g., CIDEr) between different groups. The bias ranges from 0.7% to 5.3% across all metrics, indicating a relatively low level of bias in the model.

## Scaling
The model variants of microsoft/git-base benefit significantly from more pre-training data. Scaling up the image encoder, specifically the backbone, improves the model's performance, especially with 0.8B data. However, scaling up the text decoder has shown no improvement, possibly due to the difficulty in effectively training with limited text data.

## Interpretability and Explainability
The model card mentions an experimental section on explainability/interpretability, but no specific details or code blocks are provided. [More Information Needed]

---

Please note that the above model card description is a generic description based on the provided references. The actual model card may require additional information specific to the microsoft/git-base model.

## Environmental Impact

- **Hardware Type:** The model microsoft/git-base is trained on A100 provisioned by Azure Machine Learning.
- **Software Type:** The model microsoft/git-base is trained on a generative image-to-text (GIT) scheme, where the class names are interpreted as image captions, and the model is fine-tuned to predict the result in an auto-regressive way. Unlike existing work, which typically uses a pre-defined vocabulary and a linear layer for classification, the GIT model benefits from its generation-based approach, especially when new data and categories are added to the dataset. The model's parameters are updated to better fit the visual-linguistic tasks, and it achieves state-of-the-art performance on various benchmarks.
- **Hours used:** Based on the provided references, the amount of time used to train the model microsoft/git-base is not mentioned. Therefore, the answer is "[More Information Needed]".
- **Cloud Provider:** The model microsoft/git-base is trained on Azure Machine Learning, which is provided by the cloud provider Microsoft Azure.
- **Carbon Emitted:** The amount of carbon emitted when training the model microsoft/git-base is not provided in the given references. [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The model microsoft/git-base has a generative image-to-text transformer architecture. It consists of an image encoder and a text decoder. The image encoder is based on a contrastive pre-trained model, which takes raw images as input and outputs a compact 2D feature map. This feature map is flattened into a list of features and then projected into a D-dimensional space using a linear layer and a layernorm layer. 

The text decoder is a transformer module that predicts the text description. It is composed of multiple transformer blocks, each containing a self-attention layer and a feed-forward layer. The text is tokenized and embedded into D dimensions, followed by positional encoding and a layernorm layer. The image features are concatenated with the text embeddings as the input to the transformer module. The text decoding process starts with the [BOS] token and continues in an auto-regressive manner until the [EOS] token or the maximum number of steps is reached. The seq2seq attention mask is applied to ensure that the text tokens depend only on the preceding tokens and all image tokens.

The objective of the microsoft/git-base model is to pre-train the image encoder and text decoder using a language modeling task. The model is trained on a large dataset of 0.8 billion image-text pairs. The pre-training task is to map the input image to the associated text description. The model aims to achieve state-of-the-art performance on various vision and language benchmarks.

Please note that the above description is based on the provided references, and further information may be needed for a more comprehensive understanding of the model.

### Compute Infrastructure

The compute infrastructure for the model microsoft/git-base is as follows:

The training is conducted on A100 provisioned by Azure Machine Learning. The code is written in Python and uses packages such as Pytorch, DeepSpeed, Transformers, maskrcnn-benchmark, CLIP, OSCAR, and VirTex.

[More Information Needed]

## Citation

```
@misc{jianfeng-git,
    author = {Jianfeng Wang and
              Zhengyuan Yang and
              Xiaowei Hu and
              Linjie Li and
              Kevin Lin and
              Zicheng Liu and
              Ce Liu and
              Lijuan Wang},
    title  = {GIT: A Generative Image-to-text Transformer for Vision and Language},
    url    = {https://arxiv.org/pdf/2205.14100.pdf}
}
```

