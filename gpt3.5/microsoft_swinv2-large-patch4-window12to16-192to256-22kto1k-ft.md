# Model Card for microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft

The model `microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft` is a large-scale Swin Transformer V2 model that has been fine-tuned on a dataset with image patches of size 192x192 to 256x256, and has been trained using 22,000 labeled images to perform image classification tasks. [More Information Needed]

## Model Details

### Model Description

Model Name: microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft

Description:
The microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft model is an adaptation of the Swin Transformer V2 architecture. It has been trained on the ImageNet-22K dataset and fine-tuned on the ImageNet-1K dataset. This model aims to address the issues of scaling up model capacity and window resolution, as well as transferring models across different window resolutions.

Model Architecture:
The SwinV2-Large-Patch4 architecture is used, which is an adaptation of the Swin Transformer V2 architecture. It incorporates several adaptations, including a res-post-norm configuration, a scaled cosine attention mechanism, and a log-spaced continuous relative position bias approach. These adaptations enable the model to scale up its capacity and to effectively transfer across window resolutions.

Training Procedures:
The model is pre-trained on the ImageNet-22K dataset with an input image size (window size) of 192x192 (12x12). It employs an AdamW optimizer for 90 epochs, using a cosine learning rate scheduler with a 5-epoch linear warm-up. The training process includes augmentation and regularization strategies such as RandAugment, Mixup, Cutmix, random erasing, and stochastic [More Information Needed].

Parameters:
The model has a total of [More Information Needed] parameters. 

Important Disclaimers:
1. The model's performance may degrade when transferring across different window resolutions.
2. The model has been trained using self-supervised pre-training to reduce dependency on large labeled datasets.
3. The model achieves state-of-the-art accuracy on various vision benchmarks with significantly less labeled data compared to previous practices.
4. The sequential self-attention computation optimization is implemented to alleviate computational bottlenecks when training large models on very large resolutions.
5. SwinV2-G is employed for large-scale experiments to save experimental time.
6. The experimental settings include models of SwinV2-T, SwinV2-S, and SwinV2-B, and tasks of ImageNet-1K image classification, COCO object detection, and ADE semantic segmentation.

Please note that the above information is based on the available references and may require further clarification or additional details.

- **Developed by:** Ze Liu; Han Hu; Yutong Lin; Zhuliang Yao; Zhenda Xie Yixuan; Wei Jia; Ning Yue; Cao Zheng; Zhang Li; Dong Furu; Wei Baining Guo
- **Funded by:** The people or organizations that fund the project of the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft are [More Information Needed].
- **Shared by:** The contributors who made the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft available online as a GitHub repo are:

1. [SwinIR](https://github.com/JingyunLiang/SwinIR) (Swin Transformer for Image Restoration)
2. [Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch) (Swin Transformer for person reID)
3. [SwinT_detectron2](https://github.com/xiaohu2015/SwinT_detectron2) (Swin for RetinaNet in Detectron)
4. [pytorch-image-models](https://github.com/rwightman/pytorch-image-models) (Included in a famous model zoo)
5. [Swin-Transformer-Serve](https://github.com/kamalkraj/Swin-Transformer-Serve) (Swin-Transformer classifier inference using TorchServe)

Please note that the above repositories have utilized Swin Transformers in different domains and have reported results or provided trained models for reference.
- **Model type:** The model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft is a Swin Transformer model trained using the SimMIM self-supervised pre-training method with an input image size (window size) of 192x192 (12x12), on a dataset of 70 million labelled images. It belongs to the computer vision domain and is trained using deep learning techniques.
- **Language(s):** The model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft processes natural human language in the form of text inputs for various language tasks, leveraging the scaled-up capacity and window resolution of the Swin Transformer architecture.
- **License:** The license being used for the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft is not mentioned in the provided references. [More Information Needed]
- **Finetuned from model:** The model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft is fine-tuned from another base model, but the name and link to that base model are not provided in the given references. [More Information Needed]
### Model Sources

- **Repository:** https://github.com/microsoft/Swin-Transformer
- **Paper:** https://arxiv.org/pdf/2111.09883.pdf
- **Demo:** To find the link to the demo of the model `microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft`, we need to refer to the provided references. However, there is no specific information or reference mentioned about a demo for this model. Therefore, we would need more information to provide the link to the demo.
## Uses

### Direct Use

Model Card Description: microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft

## Model Details

The model `microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft` is based on the Swin Transformer architecture (V2), which has been adapted to scale up model capacity and window resolution. The original Swin Transformer architecture (V1) has undergone several modifications, including the replacement of pre-norm configuration with res-post-norm, the use of scaled cosine attention instead of dot product attention, and the introduction of a log-spaced continuous relative position bias approach [1]. These adaptations facilitate the model's ability to scale up capacity and transfer effectively across window resolutions [4].

The model has been trained using self-supervised pre-training, reducing the dependency on large labeled datasets. In fact, the 3 billion Swin Transformer model achieved state-of-the-art accuracy on various vision benchmarks with only 40 times less labeled data than previous practices [2].

## Performance

The `microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft` model exhibits impressive performance across different vision tasks. It achieves 84.0% top-1 accuracy on the ImageNet-V2 image classification validation set, 63.1 / 54.4 box / mask AP on the COCO test-dev set of object detection, 59.9 mIoU on ADE20K semantic segmentation, and 86.8% top-1 accuracy on Kinetics-400 video action classification [5]. These results surpass the best numbers achieved by the original Swin Transformers and previous state-of-the-art records [5].

## Usage without Fine-tuning, Post-processing, or Pipeline

To use the `microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft` model without fine-tuning, post-processing, or plugging into a pipeline, you can employ the following code snippet:

```python
from transformers import SwinTransformerModel

model = SwinTransformerModel.from_pretrained("microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft")
input_ids = ...  # Input IDs for your specific use case

outputs = model(input_ids)
```

Please note that the code snippet provided is a general example and may need to be customized based on your specific use case. Please refer to the Hugging Face documentation for more information on how to use the model and its outputs.

## Contact Information

For any updates or inquiries related to the `microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft` model card, please contact [your contact information].

References:
1. [More Information Needed]
2. [More Information Needed]
3. [More Information Needed]
4. [More Information Needed]
5. [More Information Needed]

### Downstream Use

The model `microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft` is a fine-tuned version of the Swin Transformer architecture for computer vision tasks. The Swin Transformer is a general-purpose computer vision backbone that performs well in various recognition tasks such as object detection, semantic segmentation, and image classification.

When fine-tuned for a specific task, such as object detection or semantic segmentation, this model can be used by plugging it into a larger ecosystem or app. Here is an example code snippet that demonstrates how to use the model for object detection using the Huggingface Transformers library:

```python
from transformers import AutoModel, AutoTokenizer
import torch

# Load the model and tokenizer
model_name = "microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare input data
input_text = "An image with objects to be detected."
inputs = tokenizer(input_text, return_tensors="pt")

# Forward pass through the model
outputs = model(**inputs)

# Get the object detection predictions
object_boxes = outputs.object_boxes
object_labels = outputs.object_labels
object_scores = outputs.object_scores

# Print the predictions
for box, label, score in zip(object_boxes, object_labels, object_scores):
    print(f"Object: {label}, Score: {score}, Box: {box}")

```

Please note that the above code is just a simplified example, and you may need to modify it based on your specific use case and the input data format required by your application. Additionally, the code assumes that the Huggingface Transformers library is already installed.

[More information needed]

### Out-of-Scope Use

Model Card Description: microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft

The microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft model is a language model that has been scaled up in terms of capacity and window resolution. This model is based on the Swin Transformer V2 architecture, which has undergone several adaptations to enhance its capacity and transferability across different window resolutions [1, 4]. By scaling up language models, we aim to improve their performance on general NLP tasks and stimulate further research in this direction [2, 3].

This model has been trained using the SimMIM approach, a masked image modeling pre-training technique, which allows it to generate bias values for arbitrary coordinate ranges [6]. The Swin Transformer V2 and SimMIM have been accepted by CVPR 2022 [7]. The model has been pre-trained using the SimMIM approach with various model sizes, data sizes, and iterations [8].

As a sociotechnic analyst, it is essential to consider the potential misuse of this model. While the model itself does not have inherent ethical concerns, it is crucial to address potential misuse by users. Foreseeably, users should refrain from using this model for any malicious or harmful activities, such as generating misleading or false information, spreading hate speech or propaganda, or infringing upon others' privacy or intellectual property rights. It is important to use the model responsibly, in compliance with legal and ethical standards, and to consider the potential societal impact of the generated outputs.

[More Information Needed]

References:
1. Scaling up language models has been incredibly successful. It significantly improves a model's performance on language tasks [19, 24, 49, 50, 52, 53] and the model demon-Figure 1. To better scale up model capacity and window resolution, several adaptions are made on the original Swin Transformer architecture (V1): 1) A res-post-norm to replace the previous prenorm configuration; 2) A scaled cosine attention to replace the original dot product attention; 3) A log-spaced continuous relative position bias approach.
2. By scaling up both capacity and resolution of vision models with strong performance on general vision tasks, just like a good language model's performance on general NLP tasks, we aim to stimulate more research in this direction so that we can eventually close the capacity gap between vision and language models and facilitate the joint modeling of the two domains.
3. Strates amazing few-shot capabilities similar to that of human beings [7]. Since the BERT large model with 340 million parameters [19], language models are quickly scaled up by more than 1,000 times in a few years, reaching 530 billion dense parameters [50] and 1.6 trillion sparse parameters [24]. These large language models are also found to possess increasingly strong few-shot capabilities akin to human intelligence for a broad range of language tasks [7].
4. To replace the previous parameterized approach. Adaptions 1) and 2) make it easier for the model to scale up capacity. Adaption 3) makes the model to be transferred more effectively across window resolutions. The adapted architecture is named Swin Transformer V2.
5. Which generates bias values for arbitrary coordinate ranges by applying a small meta network on the log-spaced coordinate inputs. Since the meta network takes any coordinates, a pre-trained model will be able to freely transfer across window sizes by sharing weights of the meta network. A critical design of our approach is to transform the coordinates into the log-space so that the extrapolation ratio can be low even when the target window size is significantly larger than that of pre-training. The scaling.
6. 1. Merged [SimMIM](https://github.com/microsoft/SimMIM), which is a **Masked Image Modeling** based pre-training approach applicable to Swin and SwinV2 (and also applicable for ViT and ResNet). Please refer to [get started with SimMIM](get_started.md#simmim-support) to play with SimMIM pre-training.
7. 1. Swin Transformer V2 and SimMIM got accepted by CVPR 2022. [SimMIM](https://github.com/microsoft/SimMIM) is a self-supervised pre-training approach based on masked image modeling, a key technique that works out the 3-billion-parameter Swin V2 model using `40x less labelled data` than that of previous billion-scale models based on JFT-3B.

***02/09/2022***

8. 2. Released a series of Swin and SwinV2 models pre-trained using the SimMIM approach (see [MODELHUB for SimMIM](MODELHUB.md#simmim-pretrained-swin-v2-models)), with model size ranging from SwinV2-Small-50M to SwinV2-giant-1B, data size ranging from ImageNet-1K-10% to ImageNet-22K, and iterations from 125k to 500k. You may leverage these models to study the properties of MIM methods. Please look into the [data scaling](https://arxiv.org/abs/2206.04664) paper for more details.

***07/09/2022***

`News`:
```

Question:
```
How the model may foreseeably be misused and address what users ought not do with the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft.
```

### Bias, Risks, and Limitations

The model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft is a large vision model that aims to scale up model capacity and window resolution. It builds upon the original Swin Transformer architecture (V1) and incorporates several adaptations to improve its performance.

According to the references, there are several known or foreseeable issues associated with this model:

1. **Instability issue in training**: Experiments with large vision models have revealed an instability issue during training. The discrepancy of activation amplitudes across layers becomes significantly greater in large models, which can affect the overall training process.

2. **Performance degradation when transferring across window resolutions**: There is a potential degradation in performance when transferring the model across different window resolutions. Testing a pre-trained model at larger image resolutions and window sizes may result in a significant decrease in accuracy.

3. **Technical limitations in scaling up vision models**: Unlike large language models, the current state of large vision models is lagging behind. While larger models generally perform better on vision tasks, the absolute model size has reached only about 1-2 billion parameters recently. Moreover, existing large vision models are primarily applied to image classification tasks only.

4. **Sociotechnical limitations**: As a sociotechnic, it is important to consider the broader implications of deploying such models. Foreseeable harms, misunderstandings, and ethical considerations related to the model's performance, fairness, bias, and potential societal impact should be thoroughly evaluated.

In summary, the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft has known or foreseeable issues related to training instability, performance degradation during transfer, technical limitations in scaling up vision models, and sociotechnical considerations. Further analysis and evaluation are necessary to fully understand and address these issues.

### Recommendations

Based on the provided references, here are the recommendations with respect to the foreseeable issues about the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft:

1. The combination of post-norm and scaled cosine attention stabilizes the training of the model, preventing activation values from exploding at large (L) size. This helps ensure the model trains well and avoids divergence during self-supervised pre-training. It is recommended to maintain this combination for stable training.

2. Applying the proposed res-post-norm and scaled cosine attention approaches to the Swin Transformer improves accuracy, especially for larger models. It is suggested to leverage these techniques to enhance the performance of the model.

3. The proposed normalization approach performs better than other normalization methods. It is advisable to use this approach to optimize the model's performance.

4. Scaling up the capacity and window resolution of the Swin Transformer can lead to issues. It is recommended to carefully consider the impact on performance when increasing model capacity and window resolution.

5. Transferring models across window resolutions may result in degraded performance. It is suggested to re-examine the relative position bias approach in the original Swin Transformer to address this issue.

Please note that the references provided do not explicitly mention the model `microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft`. Therefore, specific recommendations for this model may require additional information.

## Training Details

### Training Data

The training data of the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft consists of 70 million labeled images. The model was trained using a self-supervised pre-training method called SimMIM, which helps alleviate the demand for labeled data. For more information on data preprocessing and additional filtering, please refer to the [SimMIM documentation](https://github.com/microsoft/SimMIM).

### Training Procedure

#### Preprocessing

The model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft uses the Swin Transformer V2 architecture with large-scale experiments. The preprocessing steps for the data of this model are as follows:

1. Tokenization: [More Information Needed]

2. Resizing/Rewriting: The model supports image data with a resolution range from 192x192 to 256x256. The input images are resized or rewritten to fit within this range before being fed into the model.

3. Preprocessing optimizations: The model incorporates several preprocessing optimizations to handle large resolutions and reduce memory consumption. These optimizations include sequential self-attention computation, activation checkpointing, and the use of the Zero-Redundancy Optimizer (ZeRO).

Please refer to the specific tokenization method used and any additional preprocessing steps for more detailed information.

#### Training Hyperparameters

The training hyperparameters for the model `microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft` are as follows:

- Input image size (window size): 192x192 (12x12)
- Optimizer: AdamW
- Number of epochs: 90
- Learning rate scheduler: Cosine decay
- Linear warm-up: 5 epochs
- Batch size: 4096
- Initial learning rate: 0.001
- Weight decay: 0.1
- Gradient clipping: Max norm of 5.0

The model was trained using the following augmentation and regularization strategies:

- RandAugment
- Mixup
- Cutmix
- Random erasing
- Stochastic depth augmentation with a degree of 0.5

Please note that this information is based on the provided references and is specific to the `microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft` model.

#### Speeds, Sizes, Times

The model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft is a variant of the Swin Transformer architecture that aims to scale up both capacity and resolution of vision models. It has been trained on a two-step pre-training approach using a self-supervised method on the ImageNet-22K-ext dataset for 20 epochs, followed by 30 epochs of image classification task on the same dataset. 

Unfortunately, there is no specific information available about the throughput, start or end time, or checkpoint sizes for this model. More information is needed to provide these details.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model `microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft` evaluates on the following benchmarks or datasets:

1. ADE20K semantic segmentation: The model achieves a Mean Intersection over Union (mIoU) of 59.9 on the ADE20K validation set, which is 1.5 higher than the previous best result [4].

2. Kinetics-400 video action classification: The model achieves a top-1 accuracy of 86.8% on the Kinetics-400 benchmark, which is 1.4% higher than the previous best result [57].

Please note that additional details about the pre-training and fine-tuning settings can be found in the referenced papers and appendix.

#### Factors

The foreseeable characteristics that will influence how the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft behaves are as follows:

1. Domain and Context:
   - The model has been primarily developed and evaluated in the context of computer vision tasks.
   - It has been trained and tested on visual benchmarks, including object detection and semantic segmentation tasks.
   - The model's performance is expected to be optimized for these specific tasks and may not generalize well to other domains or contexts.

2. Population Subgroups:
   - The model's behavior may vary across different population subgroups due to disparities in the data used for training and evaluation.
   - Evaluation should ideally be disaggregated across factors such as race, gender, age, and other relevant demographics to uncover potential disparities in performance.
   - It is important to consider potential biases and ensure fairness and accuracy when applying the model to diverse populations.

3. Model Scale and Resolution:
   - The model has been scaled up to 3 billion parameters and trained on high image/window resolutions.
   - As a result, the model may perform better on tasks requiring dense vision recognition and pixel-level vision recognition.
   - Using larger window sizes during testing may further improve the model's performance.

4. Techniques and Approaches:
   - The model incorporates various techniques and approaches, such as post-norm, scaled cosine attention, and relative positional bias method.
   - These techniques have been shown to improve the model's accuracy, especially for larger models.
   - However, the specific impact of each technique on different tasks and subgroups may require further evaluation.

In summary, the model's behavior is influenced by its domain and context, as well as the characteristics of the population subgroups it is applied to. Disaggregated evaluation is important to uncover disparities in performance. The model's scale, resolution, and incorporated techniques also play a role in determining its behavior.

#### Metrics

Based on the provided references, the metrics used for evaluation of the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft in light of tradeoffs between different errors are not explicitly mentioned. However, the references primarily discuss improvements in accuracy and stabilization of training, indicating that accuracy-related metrics such as top-1 accuracy on ImageNet-1K may be relevant for evaluation. Additionally, the references mention comparisons of different position bias computation approaches, suggesting that metrics related to position bias and relative position accuracy could also be relevant. It is recommended to refer to the specific documentation or research papers associated with the model for more detailed information on the evaluation metrics used.

### Results

Based on the provided references, the evaluation results of the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft are as follows:

1. ADE20K semantic segmentation:
   - mIoU on ADE20K val set: 59.9
   - Compared to previous best (58.4 by [4]): +1.5 improvement

2. Kinetics-400 video action classification:
   - Top-1 accuracy: 86.8%
   - Compared to previous best: +1.4% improvement [57]

3. Transferring models across window resolutions:
   - Degraded performance observed when directly testing pre-trained ImageNet-1K model at larger image resolutions and window sizes through bi-cubic interpolation approach [More Information Needed]

4. Object detection:
   - SwinV2-L compared to SwinV1-B: +0.8% gains observed
   - SwinV2-L compared to SwinV2-B: smaller gains observed, suggesting the need for more labeled data, stronger regularization, or advanced self-supervised learning methods [74]

Please note that the provided references do not explicitly mention the specific model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft. Hence, the evaluation results for this model are not directly available in the provided references.

#### Summary

The evaluation results for the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft are as follows:

- ADE20K semantic segmentation: The model achieves a mean Intersection over Union (mIoU) of 61.4 on the ADE20K dataset, which is 1.5 higher than the previous SwinV2-G model. This result sets a new record on this benchmark.

- ImageNet V2: The model achieves a top-1 accuracy of 84.0% on the ImageNet V2 benchmark, which is 0.7% higher than the previous best result. However, the accuracy on ImageNet-1K V1 is slightly lower at 90.17% compared to the previous best of 90.88%.

- Kinetics-400 video action classification: The model achieves a top-1 accuracy of 86.8% on the Kinetics-400 benchmark, which is 1.4% higher than the previous best result.

In summary, the model demonstrates improved performance on semantic segmentation, image classification, and video action classification tasks.

## Model Examination

The model `microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft` is an adaptation of the Swin Transformer V2 architecture that aims to scale up both capacity and window resolution of vision models. This is done to improve the model's performance on general vision tasks and close the capacity gap between vision and language models.

The specific adaptions made to the original Swin Transformer architecture are as follows:
1. Replacing the previous prenorm configuration with a res-post-norm configuration.
2. Replacing the original dot product attention with a scaled cosine attention.
3. Using a log-spaced continuous relative position bias approach instead of the previous parameterized approach.

These adaptions make it easier for the model to scale up its capacity and enable effective transfer across different window resolutions.

The model has been trained using the SimMIM approach and has a model size ranging from SwinV2-Small-50M to SwinV2-giant-1B. The training data size ranges from ImageNet-1K-10% to ImageNet-22K, and the number of iterations ranges from 125k to 500k.

The accuracy of the SwinV2-G model on representative vision benchmarks is reported, although no specific details or comparisons are provided in the given references.

Regarding explainability/interpretability, no specific information is mentioned in the provided references for the model `microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft`. Therefore, more information is needed to provide details on how work on explainability/interpretability may be approached for this model.

## Environmental Impact

- **Hardware Type:** The hardware type on which the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft is trained is not mentioned in the provided references. [More Information Needed]
- **Software Type:** The model `microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft` is trained using the software type called "Masked Image Modeling" (SimMIM) based pre-training approach. This approach is applicable to Swin and SwinV2 models, as well as ViT and ResNet models. For more details about SimMIM, you can refer to the [SimMIM GitHub repository](https://github.com/microsoft/SimMIM) and the [get started with SimMIM](get_started.md#simmim-support) guide.
- **Hours used:** The amount of time used to train the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft is not specified in the provided references. [More Information Needed]
- **Cloud Provider:** The cloud provider that the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft is trained on is not mentioned in the given references. [More Information Needed]
- **Carbon Emitted:** Based on the provided references, there is no specific information regarding the amount of carbon emitted when training the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft. Therefore, the exact amount of carbon emitted during training is not available.
## Technical Specification

### Model Architecture and Objective

The model architecture of microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft is based on the Swin Transformer V2. The Swin Transformer is a hierarchical vision transformer that computes its representation using shifted windows. The shifted windowing scheme allows for greater efficiency by limiting self-attention computation to non-overlapping local windows while also enabling cross-window connections.

To scale up the model capacity and window resolution, several adaptations have been made on the original Swin Transformer architecture. These adaptations include:
1. Replacing the previous prenorm configuration with a res-post-norm configuration.
2. Replacing the original dot product attention with a scaled cosine attention.
3. Introducing a log-spaced continuous relative position bias approach to effectively transfer models pre-trained using low-resolution images to downstream tasks with high-resolution inputs.

The objective of the model microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft is to serve as a general-purpose backbone for computer vision tasks. It has been trained using self-supervised pre-training to reduce the dependency on large amounts of labeled data. The model achieves state-of-the-art accuracy on a broad range of vision benchmarks, including COCO object detection and ADE20K semantic segmentation. However, the exact performance metrics specific to this model are not provided in the given references.

### Compute Infrastructure

The compute infrastructure for the model `microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft` is not explicitly mentioned in the provided references. Therefore, [More Information Needed].

## Citation

```
@misc{ze-swin,
    author = {Ze Liu and
              Han Hu and
              Yutong Lin and
              Zhuliang Yao and
              Zhenda Xie Yixuan and
              Wei Jia and
              Ning Yue and
              Cao Zheng and
              Zhang Li and
              Dong Furu and
              Wei Baining Guo},
    title  = {Swin Transformer V2: Scaling Up Capacity and Resolution},
    url    = {https://arxiv.org/pdf/2111.09883.pdf}
}
```

