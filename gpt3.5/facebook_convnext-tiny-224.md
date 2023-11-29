# Model Card for facebook/convnext-tiny-224

The model facebook/convnext-tiny-224 is a pure ConvNet model constructed entirely from standard ConvNet modules. It is accurate, efficient, scalable, and simple in design. [More Information Needed]

## Model Details

### Model Description

Model Name: facebook/convnext-tiny-224

Description:
The facebook/convnext-tiny-224 model is a ConvNeXt model that is designed to perform well on image classification, object detection, instance segmentation, and semantic segmentation tasks. It is a pure ConvNet model that competes favorably with hierarchical vision Transformers in terms of accuracy, scalability, and robustness across major benchmarks.

Model Architecture:
The model architecture of facebook/convnext-tiny-224 is based on the ConvNeXt design, which is constructed entirely from standard ConvNet modules. It maintains the efficiency of standard ConvNets and has a fully-convolutional nature for both training and testing, making it simple to use.

Training Procedures:
The training procedure of the model follows a fixed training recipe with specific hyperparameters. The exact set of hyperparameters used can be found in Appendix A.1. The training is extended to 300 epochs from the original 90 epochs for ResNets. The model uses the AdamW optimizer and incorporates data augmentation techniques such as Mixup, Cutmix, RandAugment, Random Erasing, and regularization schemes including Stochastic Depth and Label Smoothing.

Parameters:
The model has a total of [More Information Needed] parameters. 

Important Disclaimers:
The performance of the ConvNeXt model is influenced not only by the network architecture but also by the training procedure and associated hyperparameter settings. The results reported for the model are based on specific settings and may vary with different configurations. It is also important to note that while ConvNeXt performs well on various tasks, there may be cases where hierarchical vision Transformers or other models are more suitable. 

For more information on the model and its usage, please refer to the provided code and resources:

- ImageNet-1K Training Code
- ImageNet-22K Pre-training Code
- ImageNet-1K Fine-tuning Code
- Downstream Transfer (Detection, Segmentation) Code
- Image Classification Colab Notebook
- Web Demo on Hugging Face Spaces

- **Developed by:** Zhuang Liu; Hanzi Mao; Chao-Yuan Wu; Christoph Feichtenhofer; Trevor Darrell; Saining Xie; A I Facebook;  Research; U C Berkeley
- **Funded by:** [More Information Needed]
- **Shared by:** The contributors who made the model facebook/convnext-tiny-224 available online as a GitHub repo are not mentioned in the provided references. [More Information Needed]
- **Model type:** The model facebook/convnext-tiny-224 is a pure ConvNet model that is accurate, efficient, scalable, and very simple in design, and it can perform as well as a hierarchical vision Transformer on various computer vision tasks including image classification, object detection, instance and semantic segmentation.
- **Language(s):** The model facebook/convnext-tiny-224 processes natural human language in the form of research papers discussing the design and performance of ConvNeXt, a pure ConvNet model constructed entirely from standard ConvNet modules.
- **License:** The license being used for the model facebook/convnext-tiny-224 is the MIT license. You can find more information about the license in the [LICENSE](LICENSE) file.
- **Finetuned from model:** Model: facebook/convnext-tiny-224

## Description

ConvNeXt is a pure ConvNet model constructed entirely from standard ConvNet modules. It is accurate, efficient, scalable, and simple in design. ConvNeXt competes favorably with Transformers in terms of accuracy, scalability, and robustness across major benchmarks in various vision tasks such as ImageNet classification, object detection/segmentation on COCO, and semantic segmentation on ADE20K [1, 6].

## Intended Use

The model can be used for a wide range of computer vision tasks, including image classification, object detection, and semantic segmentation. It is especially useful for scenarios where efficiency and simplicity are desired, without compromising on accuracy and scalability.

## Training Data

The training data used for the ConvNeXt model includes various vision datasets, such as ImageNet, COCO, and ADE20K [1, 6].

## Evaluation Data

The model has been evaluated on standard vision benchmarks, including ImageNet, COCO, and ADE20K [1, 6].

## Training Procedure

The training procedure for the ConvNeXt model involves gradually "modernizing" the architecture from a standard ResNet to a hierarchical vision Transformer, guided by the question of how design decisions in Transformers impact ConvNets' performance [3]. The training recipe includes techniques such as Stochastic Depth and Label Smoothing [7]. However, a detailed explanation of the training procedure is not available.

## Evaluation Results

The ConvNeXt model achieves 87.8% top-1 accuracy on ImageNet and outperforms Swin Transformers on COCO detection and ADE20K segmentation [6]. However, specific performance metrics and results on other benchmarks are not provided.

## Ethical Considerations

No information regarding ethical considerations is available.

## Limitations

No specific limitations of the ConvNeXt model are mentioned.

## Caveats and Recommendations

No caveats or recommendations are mentioned for using the ConvNeXt model.

## If the model is fine-tuned from another model

[More Information Needed]
### Model Sources

- **Repository:** https://github.com/facebookresearch/ConvNeXt
- **Paper:** https://arxiv.org/pdf/2201.03545.pdf
- **Demo:** The link to the demo of the model facebook/convnext-tiny-224 is [here](https://huggingface.co/spaces/akhaliq/convnext).
## Uses

### Direct Use

To use the model `facebook/convnext-tiny-224` without fine-tuning, post-processing, or plugging into a pipeline, you can follow the training recipe and hyperparameters provided in the references. 

First, you can train the model on the ImageNet-1K dataset for 300 epochs using the AdamW optimizer with a learning rate of 4e-3. The training should include a 20-epoch linear warmup and a cosine decaying schedule afterward. The batch size should be set to 4096, and a weight decay of 0.05 should be applied. Data augmentations such as Mixup, Cutmix, RandAugment, and Random Erasing should be used for regularization. Stochastic Depth and Label Smoothing can be applied as additional regularization techniques.

Here is a code snippet that demonstrates how to use the `facebook/convnext-tiny-224` model without fine-tuning, post-processing, or plugging into a pipeline:

```python
from PIL import Image
import requests
from torchvision.transforms import functional as F
from transformers import ViTFeatureExtractor, ViTForImageClassification

# Load the pre-trained ConvNeXt model
model = ViTForImageClassification.from_pretrained("facebook/convnext-tiny-224")

# Load the feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained("facebook/convnext-tiny-224")

# Preprocess the image
url = "https://example.com/image.jpg"  # Replace with your own image URL
image = Image.open(requests.get(url, stream=True).raw)
inputs = feature_extractor(images=image, return_tensors="pt")

# Perform inference
outputs = model(**inputs)

# Get the predicted class
predicted_class = outputs.logits.argmax().item()
print(f"Predicted class: {predicted_class}")
```

Please note that this code snippet assumes you have the necessary dependencies installed and have access to the internet to download the pre-trained model and preprocess the image.

### Downstream Use

The model `facebook/convnext-tiny-224` is a ConvNeXt model that has been fine-tuned on the ImageNet-1K dataset. ConvNeXt models are constructed from standard ConvNet modules and have been shown to compete favorably with Transformers in terms of accuracy, scalability, and robustness across various computer vision tasks.

When fine-tuned for a specific task, such as image classification or object detection, the `facebook/convnext-tiny-224` model can be used as a feature extractor or as a backbone network in a larger ecosystem or app. By plugging this model into a larger pipeline, you can leverage its pre-trained features to improve the performance of your task-specific model.

To use the `facebook/convnext-tiny-224` model, you can follow these steps:

1. Install the necessary dependencies, including the Huggingface Transformers library.
2. Load the model using the model's identifier (`facebook/convnext-tiny-224`).
3. Preprocess your input data by resizing it to the appropriate resolution (224x224) and normalizing it.
4. Pass the preprocessed data through the model to obtain the extracted features.
5. Use these features as input to your downstream task-specific model or application.

Here is an example code snippet to illustrate the usage:

```python
from transformers import ConvNeXtModel, ConvNeXtTokenizer

# Load the ConvNeXt model
model = ConvNeXtModel.from_pretrained("facebook/convnext-tiny-224")
tokenizer = ConvNeXtTokenizer.from_pretrained("facebook/convnext-tiny-224")

# Preprocess and tokenize your input image
input_image = preprocess_image(image)  # [More Information Needed]
inputs = tokenizer(input_image, return_tensors="pt")

# Pass the input through the model to obtain features
outputs = model(**inputs)

# Use the extracted features for your downstream task
features = outputs.last_hidden_state
```

Note that the `preprocess_image` function should handle resizing and normalization of the input image. You will need to replace `[More Information Needed]` with the appropriate code for preprocessing the image.

Overall, the `facebook/convnext-tiny-224` model is a powerful tool that can be fine-tuned for various computer vision tasks and integrated into larger applications or ecosystems to enhance their performance in tasks such as image classification or object detection.

### Out-of-Scope Use

The model facebook/convnext-tiny-224 has the potential to be misused in several ways. One foreseeable misuse is when users deploy the model without considering the potential biases in the training data. As mentioned in reference 3, a circumspect and responsible approach to data selection is necessary to avoid biases. If users fail to address this concern, the model may perpetuate or amplify existing biases present in the training data.

Another misuse of the model could occur if users rely solely on its performance without considering the limitations of ConvNeXt models compared to other models like Transformers. Reference 1 suggests that ConvNeXt models may not exhibit the same level of robustness and fairness as Transformers. Users should be cautious and conduct further investigation to understand the robustness behavior of ConvNeXt models before deploying them in critical applications.

Furthermore, it is important for users not to overlook the environmental impact of using ConvNeXt models, particularly in their larger variants. Reference 4 highlights that investigating complex model designs, including ConvNeXt, can result in increased carbon emissions. Users should be conscious of the carbon footprint associated with deploying and training these models, and consider more environmentally-friendly alternatives.

In conclusion, users of the model facebook/convnext-tiny-224 should exercise caution to avoid perpetuating biases, conduct further investigation on its robustness behavior, and consider the environmental impact of using larger ConvNeXt models.

### Bias, Risks, and Limitations

The model facebook/convnext-tiny-224 has several known or foreseeable issues and limitations:

1. **Robustness Evaluation**: The model's robustness is evaluated on various benchmark datasets such as ImageNet-A, ImageNet-R, ImageNet-Sketch, and ImageNet-C/C. The reported mean corruption error (mCE) for ImageNet-C and corruption error for ImageNet-C indicate the model's performance under different corruptions. However, further investigation on the robustness behavior of ConvNeXt vs. Transformer is needed.

2. **Domain Generalization**: With extra ImageNet-22K data, ConvNeXt-XL demonstrates strong domain generalization capabilities on ImageNet-A/R/Sketch benchmarks. However, the model's performance on other datasets or domains may vary, and additional evaluation is required.

3. **Task Suitability**: While ConvNeXt performs well on image classification, object detection, instance and semantic segmentation tasks, it may not be as suitable for certain tasks that require cross-modal feature interactions or discretized, sparse, or structured outputs. Transformers may be more flexible in such cases.

4. **Simplicity**: ConvNeXt is designed to be accurate, efficient, scalable, and simple. However, the trade-off for simplicity may limit its performance or adaptability in certain scenarios.

5. **Carbon Emissions**: The investigation of model designs, including ConvNeXt, often necessitates larger models, which can lead to increased carbon emissions. Striving for simplicity is a motivation to address this challenge.

Overall, the foreseeable issues with the model include potential limitations in robustness, domain generalization, and suitability for certain tasks. The trade-off for simplicity and the environmental impact of model design exploration are also important considerations. [More Information Needed]

### Recommendations

Based on the information provided in the references, the recommendations with respect to the foreseeable issues about the model facebook/convnext-tiny-224 are as follows:

1. Investigate robustness behavior: Further research should be conducted to understand the robustness behavior of ConvNeXt compared to other models like Transformer. This investigation will help identify any potential vulnerabilities or limitations of the ConvNeXt model.

2. Pre-training on large-scale datasets: ConvNeXt models benefit from pre-training on large-scale datasets. Therefore, it is recommended to utilize publicly available datasets or consider acquiring additional data for pre-training to improve the model's performance.

3. Consider task suitability: While ConvNeXt performs well on various computer vision tasks, it may be more suited for certain tasks compared to Transformer models. Task-specific requirements should be considered when choosing between ConvNeXt and Transformer architectures.

4. Responsible data selection: Careful consideration should be given to data selection to avoid potential biases in the training data. Acquiring diverse and representative datasets is crucial to ensure fairness and prevent biased outcomes.

5. Strive for simplicity: In the field of visual representation learning, there is a need to balance model performance with simplicity. While more sophisticated modules may improve performance, they also contribute to increased computing resources and carbon emissions. Therefore, striving for simpler model designs is recommended.

6. Scale up ConvNeXt models: To fully understand the capabilities of ConvNeXt models, it is important to scale up both the data and model size. This will allow for a more comprehensive evaluation of ConvNeXt models in comparison to other architectures, such as Swin Transformers, particularly for tasks like object detection and semantic segmentation.

Please note that the provided recommendations are based on the given references and may require further investigation or domain-specific expertise.

## Training Details

### Training Data

The training data for the model facebook/convnext-tiny-224 is not explicitly mentioned in the provided references. For more information on the data pre-processing or additional filtering, please refer to the documentation related to the model.

### Training Procedure

#### Preprocessing

The preprocessing steps for the data of the model facebook/convnext-tiny-224 are not explicitly mentioned in the provided references. Therefore, more information is needed to answer this question.

#### Training Hyperparameters

To train the model `facebook/convnext-tiny-224`, the following training hyperparameters were used:

- Training duration: 300 epochs
- Optimizer: AdamW
- Learning rate: 4e-3
- Warmup: 20 epochs with linear warmup
- Learning rate schedule: Cosine decay
- Batch size: 4096
- Weight decay: 0.05
- Data augmentations: Mixup, Cutmix, RandAugment, Random Erasing
- Regularization techniques: Stochastic Depth and Label Smoothing
- Layer Scale: Initial value of 1e-6

Please note that the above information is based on the provided references and may be subject to change or additional details.

#### Speeds, Sizes, Times

The model card description for the model `facebook/convnext-tiny-224` is as follows:

---

Model Card for `facebook/convnext-tiny-224`

## Overview
`facebook/convnext-tiny-224` is a pure ConvNet model constructed entirely from standard ConvNet modules. It is accurate, efficient, scalable, and simple in design. The model achieves 87.8% ImageNet top-1 accuracy and outperforms Swin Transformers on COCO detection and ADE20K segmentation. This model competes favorably with Transformers in terms of accuracy and scalability while maintaining the simplicity and efficiency of standard ConvNets.

## Performance
The throughput, start or end time, checkpoint sizes, and other specific details about `facebook/convnext-tiny-224` are not explicitly mentioned in the provided references. For more detailed information, please refer to the original paper and documentation.

## References
1. [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545). CVPR 2022.
2. [More Information Needed]
3. [More Information Needed]
4. [More Information Needed]
5. [More Information Needed]
6. [More Information Needed]
7. [More Information Needed]
8. [More Information Needed]
9. [More Information Needed]
10. [More Information Needed]
11. [More Information Needed]

---

Please note that additional information is needed to provide more detailed performance metrics about the model `facebook/convnext-tiny-224`.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model `facebook/convnext-tiny-224` evaluates on the following benchmarks and datasets:

1. ImageNet-1K: The model's performance is compared with other Transformer variants (DeiT and Swin Transformers) and ConvNets (RegNets, EfficientNets, and EfficientNetsV2) in terms of accuracy-computation trade-off and inference throughputs. [1]

2. COCO (Common Objects in Context): The model is fine-tuned and evaluated on object detection and segmentation tasks using Mask R-CNN and Cascade Mask R-CNN. [3]

3. ADE20K: The model's performance is evaluated on this dataset, following the training settings used in BEiT and Swin models. [5]

4. Robustness Benchmark Datasets: The model's trained/fine-tuned classification models are directly tested on several benchmark datasets, including ImageNet-A, ImageNet-R, ImageNet-Sketch, and ImageNet-C/C. Mean corruption error (mCE) for ImageNet-C, corruption error for ImageNet-C, and top-1 Accuracy for other datasets are reported. [9]

Please note that the exact details of the evaluation metrics and results are not provided in the given references.

#### Factors

The foreseeable characteristics that will influence how the model facebook/convnext-tiny-224 behaves include:

1. Model Design: The design choices of ConvNeXt, such as the combination of convolutional and next-step connections, contribute to its performance difference compared to other models. However, the specific critical components that contribute to this difference are not clearly identified.

2. Model Robustness: ConvNeXt models have shown promising robustness behaviors, outperforming state-of-the-art robust transformer models on several benchmarks. However, further investigation is needed to understand how ConvNeXt compares to Transformer models in terms of robustness, especially in different domains and contexts.

3. Large-scale Datasets: The performance of ConvNeXt models benefits from pre-training on large-scale datasets. The impact of different datasets and their characteristics on model behavior should be explored.

4. Domain Generalization: ConvNeXt-XL has demonstrated strong domain generalization capabilities with extra ImageNet-22K data. However, it is important to evaluate the model's generalization across different domains and populations to uncover disparities in performance.

5. Evaluation Disaggregation: Disaggregating the evaluation of the ConvNeXt model across factors such as domain, context, and population subgroups is crucial to uncover any disparities or biases in its performance. This will help identify if certain groups or scenarios are disproportionately affected by the model's behavior.

Overall, further research and analysis are needed to fully understand the characteristics and behavior of the facebook/convnext-tiny-224 model in different domains, contexts, and population subgroups.

#### Metrics

Based on the provided references, it is not explicitly mentioned what metrics will be used for evaluation in light of tradeoffs between different errors for the model facebook/convnext-tiny-224. Therefore, more information is needed to determine the specific metrics used for evaluation in this context.

### Results

Based on the given references, there is no explicit mention of the evaluation results for the model facebook/convnext-tiny-224. Therefore, [More Information Needed].

#### Summary

The evaluation results for the model facebook/convnext-tiny-224 are not explicitly mentioned in the given references. Therefore, more information is needed to provide a summary of the evaluation results for this specific model.

## Model Examination

The model facebook/convnext-tiny-224 is a pure ConvNet model called ConvNeXt. It is designed to be accurate, efficient, scalable, and simple in design. The model card description does not provide any specific information about the explainability/interpretability of this model. However, it mentions that the model is constructed entirely from standard ConvNet modules.

To provide more information on the explainability/interpretability of the model, further investigation is needed. The provided references do not contain any specific details about the explainability or interpretability techniques used in the development of the ConvNeXt model. Thus, it is not possible to provide a specific answer to the question about the explainability/interpretability of the model at this time.

If you would like to learn more about the explainability/interpretability of the ConvNeXt model, I recommend referring to the original research paper by Zhuang Liu et al. titled "A ConvNet for the 2020s" published in CVPR 2022. The paper may provide more insights into the explainability/interpretability techniques employed in the model development.

## Environmental Impact

- **Hardware Type:** The hardware type that the model `facebook/convnext-tiny-224` is trained on is not mentioned in the provided references. Therefore, the information about the hardware type is not available. [More Information Needed]
- **Software Type:** The model facebook/convnext-tiny-224 is trained using standard ConvNet modules.
- **Hours used:** Based on the provided references, the amount of time used to train the model facebook/convnext-tiny-224 is not specified. [More Information Needed]
- **Cloud Provider:** The cloud provider that the model facebook/convnext-tiny-224 is trained on is not mentioned in the provided references. [More Information Needed]
- **Carbon Emitted:** The amount of carbon emitted when training the model facebook/convnext-tiny-224 is not specified in the given references. [More Information Needed]
## Technical Specification

### Model Architecture and Objective

Model Architecture:

The model architecture of facebook/convnext-tiny-224 is based on ConvNeXt, which is a pure ConvNet model constructed entirely from standard ConvNet modules. It is accurate, efficient, scalable, and simple in design. The model uses separate downsampling layers between stages, which is a modification inspired by Swin Transformers. Spatial downsampling is achieved using 2×2 conv layers with stride 2. The model also adopts larger kernel-sized convolutions, ranging from 3×3 to 11×11, with the optimal performance observed at 7×7 kernel size. The model has been trained on ImageNet-1K dataset for 300 epochs using AdamW optimizer with a learning rate of 4e-3. It incorporates various data augmentations and regularization techniques, such as Mixup, Cutmix, RandAugment, Random Erasing, Stochastic Depth, Label Smoothing, and Layer Scale.

Objective:

The objective of the facebook/convnext-tiny-224 model is to provide an accurate, efficient, and scalable ConvNet model for various computer vision tasks, including ImageNet classification, object detection/segmentation on COCO, and semantic segmentation on ADE20K. The model aims to compete favorably with Transformers in terms of accuracy, scalability, and robustness across major benchmarks. It maintains the efficiency of standard ConvNets while addressing the challenges faced by global attention designs of Transformer models. The fully-convolutional nature of the model makes it extremely simple to use for both training and testing.

### Compute Infrastructure

The compute infrastructure for the model facebook/convnext-tiny-224 is not explicitly mentioned in the provided references. Therefore, the information about the compute infrastructure is "[More Information Needed]".

## Citation

```
@misc{zhuang-a,
    author = {Zhuang Liu and
              Hanzi Mao and
              Chao-Yuan Wu and
              Christoph Feichtenhofer and
              Trevor Darrell and
              Saining Xie and
              A I Facebook and
              Research and
              U C Berkeley},
    title  = {A ConvNet for the 2020s},
    url    = {https://arxiv.org/pdf/2201.03545.pdf}
}
```

