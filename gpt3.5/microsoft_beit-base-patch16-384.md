# Model Card for microsoft/beit-base-patch16-384

The model microsoft/beit-base-patch16-384 is a vision representation model called BEIT (Bidirectional Encoder representation from Image Transformers), which is pretrained using a masked image modeling task and achieves competitive results in image classification and semantic segmentation tasks.

## Model Details

### Model Description

Model Name: microsoft/beit-base-patch16-384

Model Card Description:

The microsoft/beit-base-patch16-384 model is a self-supervised vision representation model called BEIT (Bidirectional Encoder representation from Image Transformers). It is pretrained using a masked image modeling (MIM) task in a self-supervised learning manner. The model architecture is based on the Transformer model.

The input to the model is a sequence of image patches, which are linearly projected to obtain patch embeddings. A special token [S] is prepended to the input sequence, and learnable 1D position embeddings are added to the patch embeddings. The input vectors are then fed into the Transformer encoder, which consists of multiple layers of Transformer blocks.

For downstream tasks such as image classification and semantic segmentation, task-specific layers are appended upon the pretrained BEIT model, and the parameters are fine-tuned on specific datasets.

The model benefits from the use of LayerScale and relative position bias, which improve performance on tasks like ImageNet classification and ADE20K semantic segmentation. However, for extra-large models, LayerScale is not used as it can lead to instability.

The model is compared to the standard vision Transformer (ViT) in experiments, where it is found that LayerScale and relative position bias also improve ViTs on downstream tasks.

The pretrained BEIT model is fine-tuned on downstream tasks by appending task layers. The final hidden vectors are regarded as encoded representations of the input patches. For each masked position, a softmax classifier is used to predict the corresponding visual tokens.

Important Disclaimers:
- [More Information Needed]

- **Developed by:** Hangbo Bao; Li Dong; Songhao Piao; Furu Wei
- **Funded by:** The people or organizations that fund the project of the model microsoft/beit-base-patch16-384 are Microsoft.
- **Shared by:** The contributors who made the model microsoft/beit-base-patch16-384 available online as a GitHub repo are:

1. Hangbo Bao
2. Li Dong
3. Songhao Piao
4. Furu Wei

For more information, you can refer to the references provided.
- **Model type:** The model microsoft/beit-base-patch16-384 is a pretrained vision model trained using the masked image modeling (MIM) task in a self-supervised learning manner, making it suitable for downstream tasks such as image classification and semantic segmentation.
- **Language(s):** The model microsoft/beit-base-patch16-384 uses the natural human language in the form of pretrained contextualized vector representations of input images encoded by the BEIT model, which is pretrained using the masked image modeling (MIM) task in a self-supervised learning manner.
- **License:** The license being used for the model `microsoft/beit-base-patch16-384` is not mentioned in the provided references. [More Information Needed]
- **Finetuned from model:** The model microsoft/beit-base-patch16-384 is a variant of the BEIT (Bilinear Integration Transformers) model. It is fine-tuned from a base model, but the specific name and link to the base model are not provided in the references. 

[More Information Needed]
### Model Sources

- **Repository:** https://github.com/microsoft/unilm/tree/master/beit
- **Paper:** https://arxiv.org/pdf/2106.08254.pdf
- **Demo:** Based on the given references, the link to the demo of the model `microsoft/beit-base-patch16-384` is not provided.
## Uses

### Direct Use

To use the model `microsoft/beit-base-patch16-384` without fine-tuning, post-processing, or plugging into a pipeline, you can follow these steps:

1. Install the required packages:
```python
!pip install torch==1.7.1 torchvision==0.8.2 timm==0.3.2
```

2. Download the pretrained checkpoint:
```python
import torch
import urllib

checkpoint_url = "https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D"
checkpoint_path = "beit_base_patch16_224_pt22k.pth"
urllib.request.urlretrieve(checkpoint_url, checkpoint_path)

model = torch.hub.load("microsoft/beit-base-patch16-384", "beit_base_patch16_224_pt22k", pretrained=False)
model.load_state_dict(torch.load(checkpoint_path))
model.eval()
```

Please note that this code snippet assumes that the checkpoint file is stored in the same directory as the Python script.

The model `microsoft/beit-base-patch16-384` can be directly used for tasks such as image classification and semantic segmentation without any additional fine-tuning or post-processing.

### Downstream Use

The model microsoft/beit-base-patch16-384 can be used for various downstream tasks after fine-tuning. For image classification tasks, a simple linear classifier can be appended to the pretrained BEIT model. The representations of the image patches can be aggregated using average pooling, and then fed to the softmax classifier. The category probabilities can be computed using the softmax function.

Here is a code snippet that demonstrates the fine-tuning process for image classification:

```python
from transformers import BEITModel, BEITConfig
import torch.nn as nn

# Load the pretrained BEIT model
config = BEITConfig.from_pretrained("microsoft/beit-base-patch16-384")
model = BEITModel.from_pretrained("microsoft/beit-base-patch16-384", config=config)

# Append a linear classifier for image classification
num_classes = 10  # Number of target classes
classifier = nn.Linear(config.hidden_size, num_classes)

# Replace the classification head of the model with the linear classifier
model.classifier = classifier

# Fine-tune the model on the downstream task
# [More Information Needed]
```

For semantic segmentation tasks, the pretrained BEIT model can be used as a backbone encoder. Deconvolution layers can be incorporated as a decoder to produce segmentation. The model can be fine-tuned end-to-end, similar to image classification.

Unfortunately, there is no code snippet provided for the fine-tuning process for semantic segmentation. More information is needed to provide a complete code snippet.

To use the model in a larger ecosystem or app, you can leverage the Huggingface Transformers library. The model can be loaded using the `from_pretrained` method from the `BEITModel` class. Once loaded, you can use the model for inference on new input data.

```python
from transformers import BEITModel

# Load the fine-tuned model
model = BEITModel.from_pretrained("path/to/fine_tuned_model")

# Perform inference on new input data
inputs = ...  # Input data for the specific task
outputs = model(inputs)

# Process the outputs as per the requirements of your larger ecosystem or app
```

Note that you need to replace "path/to/fine_tuned_model" with the actual path to the fine-tuned model weights.

Overall, the model microsoft/beit-base-patch16-384 can be easily fine-tuned for various tasks like image classification and semantic segmentation. The Huggingface Transformers library provides a convenient way to integrate the model into larger ecosystems or apps.

### Out-of-Scope Use

Model Card Description: microsoft/beit-base-patch16-384

## Model Details

The microsoft/beit-base-patch16-384 is a deep learning model that has been developed for self-supervised visual representation learning. It is based on the BEIT (Vision Transformer with Extra Tokens) architecture, which has shown promising results in various vision and vision-language tasks.

This model has been pretrained using a self-supervised method that involves predicting patch orderings, colorization, and rotation angles. It also incorporates the masked image modeling (MIM) task, where some patches within an image are masked, and the model predicts whether the masked patches are real or fake for each masked position.

The BEIT model overcomes the limitations of pixel-level auto-encoding for vision pretraining, allowing it to capture both short-range dependencies and high-frequency details in images.

## Intended Use

The microsoft/beit-base-patch16-384 model is intended to be used as a general-purpose vision representation learning model. It can be fine-tuned on specific downstream tasks such as image classification, object detection, and image captioning.

## Ethical Considerations

While the model has been developed with the intention of benefiting society, it is important to consider potential misuse and address what users ought not to do with it. Here are some foreseeable misuses of the model:

1. Generating Deepfakes: The model could be used to generate highly realistic fake images or videos, which can be potentially misused for deceptive purposes, such as spreading misinformation or creating non-consensual explicit content.

2. Privacy Violations: If the model is trained on sensitive or private data, it could pose a risk to individuals' privacy if the learned representations are used inappropriately.

3. Bias Amplification: Pretrained models like microsoft/beit-base-patch16-384 may inadvertently amplify biases present in the training data. Care should be taken when fine-tuning the model to ensure fairness, avoid discrimination, and mitigate biases.

4. Surveillance and Tracking: The model could be used to analyze and track individuals' activities without their consent, leading to potential privacy violations and surveillance concerns.

It is important to implement safeguards and responsible practices when using the model to mitigate these potential risks. Users should adhere to ethical guidelines, respect privacy rights, and ensure transparency and accountability in their applications.

For any help or issues related to the BEIT models, you can submit a GitHub issue or contact Li Dong (`lidong1@microsoft.com`) or Furu Wei (`fuwei@microsoft.com`).

[More Information Needed]

### Bias, Risks, and Limitations

Based on the provided references, here is the answer to the question:

The known or foreseeable issues stemming from the model microsoft/beit-base-patch16-384 include:

1. Short-distance dependency: The model suffers from short-distance dependency, which affects the performance of pixel-level auto-encoding.

2. Performance on downstream tasks: Recovering all the visual tokens harms the model's performance on downstream tasks.

3. Ablation result: The ablation result is worse compared to training vision Transformer from scratch on two tasks, indicating that prediction of visual tokens is a key ingredient for the model's performance.

4. Limited data size: Pre-training the model longer can improve performance on downstream tasks, but limited available data may hinder further improvements.

5. Technical limitations: The model's focus on short-range dependencies and high-frequency details due to pixel-level auto-encoding may limit its ability to capture long-range dependencies and contextual information.

6. Sociotechnical limitations: The model's performance and impact should be evaluated in terms of potential biases, fairness, and ethical considerations to ensure responsible deployment.

Overall, the model's performance and limitations should be carefully considered, and further research is needed to address these issues and ensure the responsible and ethical use of the model.

Please note that the information provided is based on the provided references, and further analysis and investigation may be required for a comprehensive understanding of the model's limitations and potential issues.

### Recommendations

Based on the provided references, here are the recommendations with respect to the foreseeable issues about the model microsoft/beit-base-patch16-384:

1. Blockwise masking: The model benefits from blockwise masking, especially for semantic segmentation. It is recommended to continue using blockwise masking as it improves performance on both tasks.

2. Usage of visual tokens: The prediction of visual tokens is crucial for the model's performance. It is recommended to retain the usage of visual tokens as they play a key role in the model's success.

3. Ablation studies: Conducting ablation studies can help analyze the contributions of each component in the model and understand their impact on downstream tasks. Further ablation studies can provide valuable insights and help improve the model.

4. Pre-training steps: Pre-training the model for longer durations can lead to improved performance on downstream tasks. It is recommended to consider longer pre-training steps to further enhance the model's capabilities.

5. Scaling the model: Scaling the model from base to large sizes, such as 1B or 10B, can provide greater improvements in performance, especially when labeled data for supervised pre-training is insufficient. It is suggested to explore larger model sizes for even better results.

6. Ethical considerations: As a sociotechnic, it is important to consider ethical implications and potential societal impacts of the model. Further analysis and evaluation should be conducted to ensure the model's fairness, accountability, and transparency.

7. Comparative analysis: Comparing the model with other vision models and pre-training approaches can provide valuable insights. It is recommended to compare the model's performance with other supervised pre-training methods and evaluate its strengths and weaknesses.

8. Documentation and sharing: It is crucial to document the model's development process, including the code, for transparency and reproducibility. Sharing the code and model weights can facilitate collaboration and further improvements.

[More Information Needed]

## Training Details

### Training Data

The training data for the model microsoft/beit-base-patch16-384 is not explicitly mentioned in the given references. [More Information Needed]

### Training Procedure

#### Preprocessing

The data preprocessing steps for the model microsoft/beit-base-patch16-384 involve tokenization and resizing/rewriting.

Tokenization:
The image is tokenized into a 14x14 grid of visual tokens using an "image tokenizer". The number of visual tokens is equal to the number of image patches, which is determined by reshaping the image into patches. The vocabulary size for the visual tokens is set to |V| = 8192. The specific details of the image tokenizer used in the model are not mentioned in the references, but it is stated that a publicly available image tokenizer described in [RPG + 21] is used.

Resizing/Rewriting:
The 2D image is split into a sequence of patches, similar to how a standard Transformer accepts image data. The image is reshaped into N patches, where N = HW / P^2, and P is the resolution of each patch. Each image patch is flattened into a vector and linearly projected. The patches preserve the raw pixels and are used as input for the model.

In summary, the data preprocessing for microsoft/beit-base-patch16-384 involves tokenizing the image into a grid of visual tokens using an image tokenizer and reshaping the image into patches for input to the model. Unfortunately, specific details about the image tokenizer used are not provided in the references.

#### Training Hyperparameters

To answer the question about the training hyperparameters for the model `microsoft/beit-base-patch16-384`, we need to refer to the provided references. Unfortunately, there is no direct mention of the specific training hyperparameters for this model. 

Therefore, we need more information to provide the detailed training hyperparameters for the model `microsoft/beit-base-patch16-384`.

#### Speeds, Sizes, Times

Model Card for microsoft/beit-base-patch16-384:

## Model Description

The microsoft/beit-base-patch16-384 model is a variant of BEIT (Bottleneck Enhanced Integrative Transformer) that has been pretrained on a large amount of self-supervised data and then fine-tuned on downstream tasks. It follows the network architecture of ViT (Vision Transformer) and employs a linear classifier for image classification tasks.

The model utilizes a patch-based approach, where each image is divided into patches, and each patch is encoded to obtain a final encoding vector. The likelihood of labeled data is maximized by updating the parameters of BEIT and the softmax classifier.

Proper initialization is important for stabilizing the Transformer, especially for large-scale pretraining. All parameters are randomly initialized within a small range initially. The output matrices of the self-attention module and the feed-forward network in each Transformer layer are rescaled by 1/sqrt(2*l), where l is the layer index.

The model has been pretrained using self-supervised learning and intermediate fine-tuning on the ImageNet-22k dataset. It can be further fine-tuned on downstream tasks such as image classification and semantic segmentation.

## Model Performance

The performance of the microsoft/beit-base-patch16-384 model on ImageNet-1K at resolutions 224x224 and 384x384 is as follows:

- Top-1 accuracy on ImageNet-1K at resolution 224x224: [More Information Needed]
- Top-1 accuracy on ImageNet-1K at resolution 384x384: [More Information Needed]

## Model Checkpoints

You can download the pretrained checkpoints for the microsoft/beit-base-patch16-384 model from the following links:

- [BEiT-base checkpoint](https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k_ft22k.pth?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D) (self-supervised pretrained and then intermediate fine-tuned on ImageNet-22k)

## Model Comparison

Table 4 in the reference provides the results of various model variants, including BEIT with different configurations. The performance of BEIT tends to improve as the model size increases, especially for extremely larger models. Comparing BEIT 384 with ViT 384, BEIT shows greater improvements, suggesting its effectiveness for larger models when labeled data are insufficient for supervised pre-training.

Note: For more detailed information about the throughput, start or end time, and checkpoint sizes of the microsoft/beit-base-patch16-384 model, [More Information Needed].

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model microsoft/beit-base-patch16-384 evaluates on the ILSVRC-2012 ImageNet dataset with 1k classes and 1.3M images for the image classification task. This dataset is commonly used for benchmarking image classification models.

#### Factors

The model microsoft/beit-base-patch16-384 is a vision transformer that has been extensively evaluated and analyzed.

Based on the information provided, it is evident that the model has been evaluated on image classification (ImageNet) and semantic segmentation (ADE20K) tasks. The model has undergone ablation studies to analyze the contributions of its components. The results of these ablation studies have highlighted the benefits of blockwise masking, visual tokens, and the proposed masked image modeling task.

The model has been compared with other models and has achieved state-of-the-art performance on ImageNet top-1 accuracy without using extra data. The results indicate that the prediction of visual tokens is a crucial factor for the model's performance.

The model's behavior is influenced by various factors, including the domain and context of the input data. Its performance should be evaluated across different population subgroups to uncover any potential disparities in its performance.

Unfortunately, the provided information does not specify whether the evaluation has been disaggregated across factors or identifies any specific disparities in performance. Therefore, it is not possible to provide specific information about the model's behavior with regards to domain, context, and population subgroups.

For more detailed information, please refer to the references provided or contact the authors directly.

#### Metrics

The model card description for the model microsoft/beit-base-patch16-384 can be as follows:

---

## Model Card: microsoft/beit-base-patch16-384

### Description

The microsoft/beit-base-patch16-384 model is a transformer-based model that has been pre-trained on a large amount of unlabeled image data. It utilizes a novel pre-training task called masked image modeling, where it predicts the masked patches in an image. This approach outperforms naive pixel-level auto-encoding and helps the model learn to distinguish semantic regions using self-attention heads.

### Performance

The performance of the microsoft/beit-base-patch16-384 model has been evaluated on both image classification (ImageNet) and semantic segmentation (ADE20K) tasks. The model has shown significant improvements over the baseline vision Transformer (ViT) models. The ablation studies indicate that the prediction of visual tokens is a key factor for the model's performance. Blockwise masking is also beneficial, especially for pixel-level auto-encoding tasks.

### Metrics for Evaluation

The specific metrics used for evaluation in light of tradeoffs between different errors are not explicitly mentioned in the references provided. [More Information Needed]

---

Note: The answer provided is based on the information provided in the given references. If more information is available, a more detailed answer can be provided.

### Results

Based on the available references, the evaluation results of the model microsoft/beit-base-patch16-384 are not explicitly mentioned. We don't have specific information about the factors and metrics used to evaluate this model. Therefore, we need more information to provide the evaluation results for the model.

#### Summary

The evaluation results for the model microsoft/beit-base-patch16-384 are summarized as follows:

The model variants were evaluated on image classification (ImageNet) and semantic segmentation (ADE20K) tasks. Ablating blockwise masking by randomly sampling masked positions showed that blockwise masking is beneficial for both tasks, particularly for semantic segmentation. Additionally, ablation of visual tokens and predicting raw pixels of masked patches resulted in significantly worse performance compared to naive pixel-level auto-encoding.

Furthermore, the prediction of visual tokens was found to be the key ingredient of BEIT, as ablation results were worse than training vision Transformer from scratch on both tasks. Ablating both visual tokens and blockwise masking together showed that blockwise masking is even more helpful for pixel-level auto-encoding, relieving the suffering of short-distance dependency. However, recovering all visual tokens harmed performance on downstream tasks.

The evaluation was conducted through ablation studies, where the default pre-training steps were set to 300 epochs, which is 37.5% of the total steps used in previous experiments.

The model used average pooling to aggregate hidden states of each image patch and added a probing layer at the middle layer of the Transformer. The best layer for BEIT-base was found to be the 9th layer, while for BEIT-large, it was the 14th layer. The linear probe layer was updated using AdamW for 50 epochs with a learning rate of 4e-3 and cosine decay. The batch size was set to 1024, and the weight decay was 1e-4.

The evaluation results showed that BEIT-large achieved state-of-the-art ImageNet top-1 accuracy of 88.6% without the use of extra data other than ImageNet-22k. However, specific details about the evaluation metrics or performance of the model microsoft/beit-base-patch16-384 are not provided in the given references.

[More Information Needed]

## Model Examination

The model microsoft/beit-base-patch16-384 utilizes a self-attention mechanism to learn semantic regions within an image without any task-specific supervision. This ability allows the model to distinguish objects and improve generalization on small-scale datasets. The self-attention map, generated using attention scores computed via query-key product in the last layer, demonstrates the model's capability to attend to different patches based on reference points within the image.

Additionally, the model employs a masked image modeling (MIM) task for pre-training. The MIM task involves predicting discrete visual tokens to summarize high-level abstractions, which outperforms pixel-level auto-encoding in terms of capturing long-range dependencies and avoiding a focus on high-frequency details.

Ablation studies conducted on various model variants indicate the benefits of blockwise masking and the usage of visual tokens. Blockwise masking, where positions are randomly masked, proves to be particularly advantageous for semantic segmentation. Predicting the raw pixels of masked patches as a pixel regression problem is less effective compared to the proposed masked image modeling task.

In terms of explainability/interpretability, the model's self-attention mechanism plays a crucial role. The visualizations of self-attention maps demonstrate how the model attends to different patches, allowing for understanding and interpretation of the model's decision-making process within an image.

Overall, the model's ability to distinguish semantic regions, separate objects, and capture long-range dependencies without manual annotation makes it a valuable asset for downstream tasks and improves generalization, especially on small-scale datasets. [More Information Needed]

## Environmental Impact

- **Hardware Type:** The hardware type used to train the model microsoft/beit-base-patch16-384 is not mentioned in the given references. [More Information Needed]
- **Software Type:** The model microsoft/beit-base-patch16-384 is trained on the masked image modeling (MIM) task, which is a self-supervised learning task. It aims to recover the masked image patches based on encoding vectors. The software type used for training the model is not mentioned in the provided information. [More Information Needed]
- **Hours used:** The amount of time used to train the model microsoft/beit-base-patch16-384 is not mentioned in the provided references. [More Information Needed]
- **Cloud Provider:** The cloud provider that the model microsoft/beit-base-patch16-384 is trained on is not mentioned in the provided references. [More Information Needed]
- **Carbon Emitted:** Based on the provided references, the amount of carbon emitted when training the model microsoft/beit-base-patch16-384 is not mentioned. Therefore, the information regarding the carbon emissions during training is not available. [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The model architecture of microsoft/beit-base-patch16-384 is based on the BEIT (Bidirectional Encoder representation from Image Transformers) framework. It utilizes a standard Transformer architecture to process image data. The input image is split into a sequence of patches, which are then linearly projected and flattened into vectors. These image patches are treated as input features in BEIT.

The model objective is to perform self-supervised pre-training on image data using a masked image modeling (MIM) task. In this task, some image patches are randomly masked and fed into the backbone Transformer. The objective is to recover the original visual tokens based on the corrupted image patches. After pre-training, the model parameters are fine-tuned on downstream tasks such as image classification and semantic segmentation.

Overall, the model architecture involves processing image data using the BEIT framework and training it in a self-supervised manner with the MIM task. The goal is to learn contextualized vector representations of images that can be used for various downstream tasks.

[More Information Needed]

### Compute Infrastructure

Based on the references provided, the compute infrastructure information about the model microsoft/beit-base-patch16-384 is not explicitly mentioned. Therefore, we need more information to provide the compute infrastructure details for this model.

## Citation

```
@misc{hangbo-beit,
    author = {Hangbo Bao and
              Li Dong and
              Songhao Piao and
              Furu Wei},
    title  = {BEIT: BERT Pre-Training of Image Transformers},
    url    = {https://arxiv.org/pdf/2106.08254.pdf}
}
```

