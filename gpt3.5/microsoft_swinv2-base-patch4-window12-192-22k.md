# Model Card for microsoft/swinv2-base-patch4-window12-192-22k

The model microsoft/swinv2-base-patch4-window12-192-22k is a Swin Transformer V2 model that has been pretrained using the SimMIM approach. It is a self-supervised pre-training approach based on masked image modeling and is able to achieve the performance of a 3-billion-parameter model using 40x less labeled data. For more details, please refer to the [SimMIM](https://github.com/microsoft/SimMIM) repository.

## Model Details

### Model Description

Model Card Description: microsoft/swinv2-base-patch4-window12-192-22k

## Basic Details

- Model Name: microsoft/swinv2-base-patch4-window12-192-22k
- Model Architecture: Swin Transformer V2
- Training Procedures: The model was trained using self-supervised pre-training (SimMIM) to reduce the dependency on labeled data. The training involved the use of residual-post-norm method combined with cosine attention to improve training stability. It also utilized a log-spaced continuous position bias method to effectively transfer models pre-trained using low-resolution images to downstream tasks with high-resolution inputs.
- Parameters: The model has 3 billion parameters, making it the largest dense vision model to date. The image resolution used for training was up to 1,536×1,536.
- Important Disclaimers: 
  - The model's performance may degrade when transferring it across different window resolutions. Further investigation is needed to address this issue.
  - The self-attention module can become a bottleneck when training large models on very high resolutions. Sequential self-attention computation was implemented to alleviate this problem.
  - Activation checkpointing technology was used to reduce GPU memory consumption during training, but it may result in slower training speed.

Note: For more detailed information, please refer to the provided references.

[More Information Needed]

- **Developed by:** Ze Liu; Han Hu; Yutong Lin; Zhuliang Yao; Zhenda Xie Yixuan; Wei Jia; Ning Yue; Cao Zheng; Zhang Li; Dong Furu; Wei Baining Guo
- **Funded by:** The model card does not provide any explicit information about the funding for the project of the model microsoft/swinv2-base-patch4-window12-192-22k. Therefore, we can conclude that the information about the people or organizations that funded this project is not available in the provided references.
- **Shared by:** The contributors who made the model microsoft/swinv2-base-patch4-window12-192-22k available online as a GitHub repo are Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Wei Jia, Ning Yue, Cao Zheng, Zhang Li, Dong Furu, and Wei Baining Guo.
- **Model type:** The model microsoft/swinv2-base-patch4-window12-192-22k is a vision model trained using self-supervised pre-training and Swin Transformer V2 architecture, with the ability to handle high-resolution input images and large attention windows. [More Information Needed]
- **Language(s):** The model microsoft/swinv2-base-patch4-window12-192-22k uses or processes natural human language to improve the performance on language tasks by scaling up model capacity and window resolution and making adaptations to the original Swin Transformer architecture.
- **License:** The license information for the model microsoft/swinv2-base-patch4-window12-192-22k is not provided in the references. [More Information Needed]
- **Finetuned from model:** The model microsoft/swinv2-base-patch4-window12-192-22k is not fine-tuned from another model. [More Information Needed]
### Model Sources

- **Repository:** https://github.com/microsoft/Swin-Transformer
- **Paper:** https://arxiv.org/pdf/2111.09883.pdf
- **Demo:** The link to the demo of the model microsoft/swinv2-base-patch4-window12-192-22k is currently not provided in the references. [More Information Needed]
## Uses

### Direct Use

Based on the available information, it is not explicitly mentioned how the model `microsoft/swinv2-base-patch4-window12-192-22k` can be used without fine-tuning, post-processing, or plugging into a pipeline. Therefore, more information is needed to provide a specific answer to this question.

### Downstream Use

Model Card Description for microsoft/swinv2-base-patch4-window12-192-22k:

## Model Details

The microsoft/swinv2-base-patch4-window12-192-22k model is a variant of the Swin Transformer architecture. It has been fine-tuned for specific tasks and can be used in various scenarios such as image classification, object detection, video action recognition, semantic segmentation, and more.

## Intended Use

This model can be used for fine-tuning on different computer vision tasks, such as image classification, object detection, and semantic segmentation. It can also be integrated into larger ecosystems or applications that require visual understanding capabilities.

## Training Data

The model has been trained on various datasets and tasks, including ImageNet-1K, COCO object detection, ADE20K semantic segmentation, and Kinetics-400 video action recognition.

## Evaluation Data

The model's performance has been evaluated on benchmark datasets relevant to the tasks it has been fine-tuned for. For example, it achieves a mIoU of 61.4 on the ADE20K semantic segmentation dataset.

## Training Procedure

The model has undergone a two-stage fine-tuning process with different hyperparameters and optimization strategies specific to each task. The training details for each task can be found in the provided references.

## Code Snippet

```python
# Example code snippet for using microsoft/swinv2-base-patch4-window12-192-22k for image classification
from transformers import SwinTransformerModel

model = SwinTransformerModel.from_pretrained("microsoft/swinv2-base-patch4-window12-192-22k")
image = load_image("path/to/image.jpg")
predictions = model.predict(image)
```

## Limitations and Known Issues

Some limitations and known issues include potential instability issues when scaling up the model capacity and the need for fine-tuning on certain tasks. For more specific limitations and known issues, please refer to the provided references.

## Ethical Considerations

As an AI model, microsoft/swinv2-base-patch4-window12-192-22k inherits the ethical considerations associated with computer vision models. These considerations include potential biases in the training data, privacy concerns related to the use of visual data, and the impact of AI systems on society. It is important to evaluate and mitigate these ethical considerations when using the model in real-world applications.

## Contributors

- [More Information Needed]

## Acknowledgments

We would like to acknowledge the contributions and references provided by the Swin Transformer community, as well as the developers and researchers who have implemented Swin Transformer for various tasks and shared their work.

### Out-of-Scope Use

Model Card Description: microsoft/swinv2-base-patch4-window12-192-22k

The microsoft/swinv2-base-patch4-window12-192-22k model is a Swin Transformer V2 model that has been trained with self-supervised pre-training using the SimMIM approach. It is a vision model with a capacity of 3 billion parameters and has shown state-of-the-art accuracy on a wide range of vision benchmarks. The model has been successfully scaled up in both capacity and resolution, with image resolution as large as 1,536x1,536.

The model's performance has been improved through various adaptations made to the original Swin Transformer architecture. These adaptations include a res-post-norm configuration, a scaled cosine attention mechanism, and a log-spaced continuous relative position bias approach. These changes make it easier to scale up the model's capacity and improve its transferability across different window resolutions.

It is important to note that the model's training relies on self-supervised pre-training, which reduces the dependency on super-huge labeled data. With 40x less labeled data than previous models, the 3 billion Swin Transformer model achieves impressive accuracy.

The microsoft/swinv2-base-patch4-window12-192-22k model is intended to stimulate further research in the field of vision models and facilitate the joint modeling of vision and language domains. By closing the capacity gap between vision and language models, it aims to encourage advancements in both domains.

As for potential misuse, it is important to consider the ethical implications of using the model. The model should not be used for any malicious or harmful purposes, such as generating or spreading misinformation, creating deepfakes, or invading privacy. It is the responsibility of users to ensure that the model's capabilities are utilized in a responsible and ethical manner. 

[More Information Needed]

### Bias, Risks, and Limitations

The foreseeable issues stemming from the model microsoft/swinv2-base-patch4-window12-192-22k include:

1. **Misunderstandings**: There may be misunderstandings regarding the capabilities and limitations of the model. Users might expect the model to perform well on tasks outside its intended use case or have unrealistic expectations about its performance.

2. **Technical Limitations**: The model may have technical limitations such as reduced performance when transferring models across different window resolutions. This could lead to degraded accuracy when testing the model on larger image resolutions and window sizes.

3. **Sociotechnical Limitations**: As a sociotechnic, it is important to consider the broader societal implications of deploying this model. The use of self-supervised pre-training approaches like masked image modeling raises concerns about data privacy, potential biases in the training data, and the impact on labor dynamics in data annotation.

4. **Foreseeable Harms**: There may be potential harms associated with the model's performance. For example, if the model is used in critical applications such as healthcare or autonomous vehicles, any inaccuracies or biases in its predictions could have serious consequences.

Overall, it is crucial to communicate these limitations and potential harms to users and stakeholders to ensure responsible and informed use of the model. Further analysis and evaluation are necessary to fully understand and address these issues.

### Recommendations

The model microsoft/swinv2-base-patch4-window12-192-22k has addressed several foreseeable issues and made recommendations based on the references provided.

1. To handle the variable window sizes between pretraining and fine-tuning, the model proposes a log-spaced continuous position bias approach (Log-CPB). This approach smoothly transfers pre-trained model weights at low resolution to deal with higher resolution windows. This recommendation helps to improve the accuracy when transferring models across window resolutions.

2. The combination of post-norm and scaled cosine attention has been shown to stabilize the training of the model. This is important because the activation values at deeper layers in the original Swin Transformer tend to explode at large sizes. The new version of the model exhibits milder behavior, which helps in training on huge size models and improves self-supervised pre-training.

3. The proposed res-post-norm and scaled cosine attention approaches have been applied to the Swin Transformer and have shown improved accuracy at different model sizes (tiny, small, and base). These techniques are particularly beneficial for larger models. Additionally, these approaches have also been found to benefit the ViT architecture.

4. The proposed log-spaced continuous position bias (CPB) approach performs marginally better than the linear-spaced approach. The benefit of the log-spaced CPB approach increases with larger changes in resolutions between pre-training and fine-tuning.

Based on the references, these are the recommendations and insights provided regarding the foreseeable issues for the model microsoft/swinv2-base-patch4-window12-192-22k.

## Training Details

### Training Data

The training data for the model microsoft/swinv2-base-patch4-window12-192-22k is not explicitly mentioned in the provided references. [More Information Needed]

### Training Procedure

#### Preprocessing

The preprocessing steps for the data used in the model microsoft/swinv2-base-patch4-window12-192-22k involve tokenization and resizing/rewriting.

Tokenization: The specific details of tokenization for this model are not mentioned in the provided references. [More Information Needed]

Resizing/Rewriting: The model operates on image data, and it is mentioned that the original Swin Transformer architecture uses a window-based approach. The window size for this model is 12x12, and the reference suggests that the model can be tested on larger image resolutions and window sizes using bi-cubic interpolation. However, the specific resizing/rewriting techniques utilized in the preprocessing are not described in detail. [More Information Needed]

In summary, more information is needed regarding the tokenization and resizing/rewriting techniques used in the preprocessing for the data of the model microsoft/swinv2-base-patch4-window12-192-22k.

#### Training Hyperparameters

The training hyperparameters for the model `microsoft/swinv2-base-patch4-window12-192-22k` are as follows:

- Input image size (window size): 192×192 (12×12)
- Optimizer: AdamW
- Number of epochs: 90
- Learning rate scheduler: Cosine learning rate decay with 5-epoch linear warm-up
- Batch size: 4096
- Initial learning rate: 0.001
- Weight decay: 0.1
- Gradient clipping max norm: 5.0
- Augmentation and regularization strategies: RandAugment, Mixup, Cutmix, random erasing, and stochastic depth

Please note that the specific details of augmentation and regularization strategies are not provided in the references, so further information is needed if you require more specific details about these strategies.

#### Speeds, Sizes, Times

The model microsoft/swinv2-base-patch4-window12-192-22k is a 3 billion-parameter Swin Transformer V2 model, which is the largest dense vision model to date. It has achieved impressive results on various representative visual benchmarks. 

Specifically, on the ImageNet-V2 image classification validation set, it obtained 84.0% top-1 accuracy. For COCO object detection, it achieved 63.1 / 54.4 box / mask AP on the test-dev set. On ADE20K semantic segmentation, it achieved a mIoU of 59.9. And for Kinetics-400 video action classification, it achieved 86.8% top-1 accuracy. These results are higher than the best numbers achieved by the original Swin Transformers.

In terms of model architecture, the SwinV2-G variant was employed for large-scale experiments. It has a channel size (C) of 512 and the number of blocks is {2, 2, 42, 4}.

The model uses a residual-post-norm method combined with cosine attention to improve training stability. Additionally, a log-spaced continuous position bias method is used to effectively transfer models pre-trained using low-resolution images to downstream tasks with high-resolution inputs. It also employs a self-supervised pretraining method called SimMIM, which reduces the need for vast labeled images.

Regarding throughput, start or end time, checkpoint sizes, and other specific details about the model, [More Information Needed].

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model microsoft/swinv2-base-patch4-window12-192-22k evaluates on the following benchmarks and datasets:

1. ADE20K semantic segmentation: The model achieves a mean intersection over union (mIoU) of 59.9% on the ADE20K validation set, which is 1.5% higher than the previous best result. [Reference 1]

2. Kinetics-400 video action classification: The model achieves a top-1 accuracy of 86.8% on the Kinetics-400 benchmark, which is 1.4% higher than the previous best result. [Reference 3]

3. ImageNet V2 image classification: The model achieves a top-1 accuracy of 84.0% on the ImageNet V2 benchmark, which is 0.7% higher than the previous best result. [Reference 4]

Please note that the model may have been evaluated on other benchmarks or datasets, but this information is not provided in the given references.

#### Factors

The foreseeable characteristics that will influence how the model microsoft/swinv2-base-patch4-window12-192-22k behaves are as follows:

1. Log-spaced continuous position bias approach: The model utilizes a log-spaced continuous position bias approach (Log-CPB) to smoothly transfer pre-trained model weights at low resolution to higher resolution windows. This approach improves the accuracy of the model.

2. Post-norm and scaled cosine attention: The combination of post-norm and scaled cosine attention stabilizes the training process. It prevents activation values from exploding, especially in deeper layers, when the model is used at large sizes.

3. Relative position bias method: In computer vision, the relative positional bias method is commonly used because spatial relationships play a crucial role in visual modeling. The model leverages this method to learn bias values as model weights, which helps improve performance.

4. Ablation on res-post-norm and scaled cosine attention: The proposed techniques of res-post-norm and scaled cosine attention positively impact the model's accuracy. These techniques are more beneficial for larger models and also benefit the ViT architecture.

5. Scaling up model capacity and window resolution: The model aims to explore the feasibility of scaling up model capacity and window resolution and whether vision tasks can benefit from significantly larger capacity. The model's accuracy is reported on representative vision benchmarks.

6. Transfer of models across window resolutions: The model may experience degraded performance when transferring models across different window resolutions. Directly testing a pre-trained ImageNet-1K model at larger image resolutions and window sizes through bi-cubic interpolation approaches leads to a decrease in accuracy. The relative position bias approach used in the original Swin Transformer should be re-examined in this context.

It is important to evaluate the model's performance by disaggregating the results across various factors such as domain, context, and population subgroups. This evaluation can help uncover disparities in performance and ensure fairness and equity in the model's behavior.

#### Metrics

According to the provided references, there is no specific mention of the metrics that will be used for evaluation in light of tradeoffs between different errors for the model microsoft/swinv2-base-patch4-window12-192-22k. Hence, more information is needed to determine the specific metrics used for evaluation in this context.

### Results

Based on the provided references, the model `microsoft/swinv2-base-patch4-window12-192-22k` has been evaluated on multiple benchmarks. However, the specific evaluation results of this model are not mentioned in the given references. Therefore, additional information is needed to provide the evaluation results of the model based on the Factors and Metrics.

#### Summary

The evaluation results for the model microsoft/swinv2-base-patch4-window12-192-22k are as follows:

1. ADE20K semantic segmentation: The model achieves a mean intersection over union (mIoU) of 59.9 on the ADE20K validation set, which is 1.5 higher than the previous best result of 58.4. Using a larger window size at test time can bring additional gains of 0.2.

2. Object detection: There is no specific information available regarding the performance of the model on object detection tasks.

3. Kinetics-400 video action classification: The model achieves a top-1 accuracy of 86.8% on the Kinetics-400 benchmark, which is 1.4% higher than the previous best result. Using a larger window size at test time can bring additional benefits of 0.2%.

4. ImageNet V2 benchmark: The model achieves a top-1 accuracy of 84.0%, which is 0.7% higher than the previous best result. However, it should be noted that the accuracy on ImageNet-1K V1 is marginally lower at 90.17% compared to 90.88% of the previous best result.

In summary, the model microsoft/swinv2-base-patch4-window12-192-22k performs well on semantic segmentation and video action classification tasks, showing improvements over previous best results. However, more information is needed regarding its performance on object detection tasks.

## Model Examination

The model microsoft/swinv2-base-patch4-window12-192-22k is an adaptation of the Swin Transformer called Swin Transformer V2. It incorporates several techniques to improve training stability and performance.

According to reference 1, the combination of post-norm and scaled cosine attention in the Swin Transformer V2 stabilizes the training process, preventing activation values from exploding at larger model sizes. This is shown in Figure 2.

Reference 2 mentions the use of a log-spaced position bias approach in the Swin Transformer V2. This approach is marginally better than the linear-spaced version and provides larger benefits when there is a significant change in resolutions between pre-training and fine-tuning.

Ablation experiments (reference 3) on the Swin Transformer with res-post-norm and scaled cosine attention techniques show improvements in accuracy for all model sizes. The techniques are more beneficial for larger models and also benefit the ViT architecture.

The model also utilizes the Zero-Redundancy Optimizer (ZeRO) (reference 4) to reduce GPU memory consumption by splitting and distributing model parameters and optimization states to multiple GPUs.

Activation check-pointing (reference 5) is another technique used in the model to reduce memory consumption of feature maps in Transformer layers. This helps to alleviate memory bottlenecks when working with high-resolution images and windows.

Reference 6 provides information about pre-trained Swin and SwinV2 models using the SimMIM approach, which can be leveraged to study the properties of MIM methods. The model microsoft/swinv2-base-patch4-window12-192-22k is part of this series of pre-trained models.

In summary, the Swin Transformer V2 adaptation in the model microsoft/swinv2-base-patch4-window12-192-22k incorporates techniques such as post-norm and scaled cosine attention, log-spaced position bias, res-post-norm, ZeRO optimizer, and activation check-pointing to improve training stability, performance, and memory efficiency.

[More Information Needed]

## Environmental Impact

- **Hardware Type:** The model microsoft/swinv2-base-patch4-window12-192-22k is trained on Nvidia A100-40G GPUs.
- **Software Type:** The model `microsoft/swinv2-base-patch4-window12-192-22k` is trained using the **Swin Transformer V2** architecture. The Swin Transformer V2 is a hierarchical Transformer that utilizes shifted windows for computing representations. It is designed as a general-purpose backbone for computer vision tasks. The model employs techniques for scaling up capacity and resolution, including res-post-norm and scaled cosine attention, as well as a log-spaced continuous relative position bias approach. 

Based on the provided information, the software type used for training the model is **Swin Transformer V2**.
- **Hours used:** Based on the provided information, the amount of time used to train the model microsoft/swinv2-base-patch4-window12-192-22k is not explicitly mentioned. Therefore, we need more information to determine the exact training time for this model.
- **Cloud Provider:** The cloud provider that the model microsoft/swinv2-base-patch4-window12-192-22k is trained on is not mentioned in the provided references. [More Information Needed]
- **Carbon Emitted:** The amount of carbon emitted when training the model microsoft/swinv2-base-patch4-window12-192-22k is not mentioned in the provided references. [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The model architecture and objective of the model microsoft/swinv2-base-patch4-window12-192-22k are as follows:

Model Architecture:
The model architecture is based on the Swin Transformer, which is a hierarchical Transformer designed for computer vision tasks. It uses shifted windows to compute representations efficiently, limiting self-attention computation to non-overlapping local windows while allowing for cross-window connections. The adapted architecture, named Swin Transformer V2, includes several modifications such as a res-post-norm configuration, scaled cosine attention, and a log-spaced continuous relative position bias approach.

Objective:
The objective of the model microsoft/swinv2-base-patch4-window12-192-22k is to serve as a general-purpose backbone for computer vision tasks. It has been trained on the ImageNet-22K dataset and is capable of processing images of size 384x384 with a window size of 24x24. The model aims to achieve high accuracy and strong performance on various granular recognition tasks, including region-level object detection, pixel-level semantic segmentation, and image-level image classification.

For more information, you can refer to the original Swin Transformer paper mentioned in Reference 7: [arxiv](https://arxiv.org/abs/2103.14030).

### Compute Infrastructure

The compute infrastructure for the model microsoft/swinv2-base-patch4-window12-192-22k is not explicitly mentioned in the given references. Therefore, we need more information to provide the compute infrastructure details for this model.

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

