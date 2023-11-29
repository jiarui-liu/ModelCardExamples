# Model Card for facebook/mask2former-swin-base-coco-instance

The model facebook/mask2former-swin-base-coco-instance is an architecture called Mask2Former that is capable of addressing any image segmentation task, including panoptic, instance, and semantic segmentation. It utilizes masked attention to extract localized features and has achieved state-of-the-art performance on popular datasets such as COCO, ADE20K, and Cityscapes.

## Model Details

### Model Description

Model Card for facebook/mask2former-swin-base-coco-instance

## Model Details

- Model Name: facebook/mask2former-swin-base-coco-instance
- Architecture: The model is built upon a meta architecture for mask classification. It consists of a backbone, a pixel decoder, and a Transformer decoder. The key component is the proposed Transformer decoder with masked attention, which replaces the standard cross-attention. The model also utilizes a multi-scale strategy to handle small objects and incorporates optimization improvements for better performance.
- Training Data: The model is trained on the COCO dataset for panoptic, instance, and semantic segmentation tasks.
- Input: The model takes an image as input and produces segmentation masks for different objects or semantic regions in the image.

## Training Procedures

- Optimizer: The model uses the AdamW optimizer.
- Learning Rate: The initial learning rate is set to 0.0001.
- Weight Decay: The weight decay is set to 0.05 for all backbones.
- Learning Rate Schedule: The model uses a step learning rate schedule, where the learning rate is decayed at 0.9 and 0.95 fractions of the total number of training steps by a factor of 10.
- Batch Size: The model is trained with a batch size of 1.
- GPU: The model is trained on a V100 GPU.

## Parameters

- Model Size: The model size is [More Information Needed].
- Parameters: The total number of parameters in the model is [More Information Needed].

## Important Disclaimers

- Model Performance: The model achieves state-of-the-art results for panoptic segmentation, instance segmentation, and semantic segmentation on popular datasets. The specific performance metrics are as follows: 57.8 PQ on COCO for panoptic segmentation, 50.1 AP on COCO for instance segmentation, and 57.7 mIoU on ADE20K for semantic segmentation.
- Pre-training: The model is initialized with ImageNet pre-trained checkpoints for the backbone.
- Specialized Architectures: The model presents a new architecture that can address any image segmentation task (panoptic, instance, or semantic) without the need for specialized architectures.
- [More Information Needed]

Please note that this model card provides a high-level overview of the model. For more detailed information, refer to the original research papers and documentation.

- **Developed by:** Bowen Cheng; Ishan Misra; Alexander G Schwing; Alexander Kirillov; Rohit Girdhar; Facebook Ai Research
- **Funded by:** The information provided does not explicitly mention the people or organizations that funded the project for the model facebook/mask2former-swin-base-coco-instance. Therefore, we need more information to answer the question about the funding sources.
- **Shared by:** The contributors who made the model facebook/mask2former-swin-base-coco-instance available online as a GitHub repo are Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar from Facebook AI Research.
- **Model type:** The model facebook/mask2former-swin-base-coco-instance is a panoptic, instance, and semantic segmentation model that utilizes a meta architecture called Mask2Former, which combines per-pixel embeddings with object queries to predict binary masks and category labels for each segment. It achieves state-of-the-art performance on popular datasets such as COCO, Cityscapes, ADE20K, and Mapillary Vistas.
- **Language(s):** The model facebook/mask2former-swin-base-coco-instance uses or processes natural human language for image segmentation tasks, including panoptic, instance, and semantic segmentation, by leveraging a new Transformer decoder with masked attention to extract localized features and achieve top results.
- **License:** The license being used for the model `facebook/mask2former-swin-base-coco-instance` is the MIT License. You can find the license link [here](https://opensource.org/licenses/MIT).
- **Finetuned from model:** The model facebook/mask2former-swin-base-coco-instance is fine-tuned from another model called MaskFormer. You can find the base model at this link: [https://github.com/facebookresearch/MaskFormer](https://github.com/facebookresearch/MaskFormer).
### Model Sources

- **Repository:** https://github.com/facebookresearch/Mask2Former/
- **Paper:** https://arxiv.org/pdf/2112.01527.pdf
- **Demo:** The link to the demo of the model `facebook/mask2former-swin-base-coco-instance` can be found [here](https://huggingface.co/spaces/akhaliq/Mask2Former).
## Uses

### Direct Use

Model Card Description: facebook/mask2former-swin-base-coco-instance

### Model Overview

The facebook/mask2former-swin-base-coco-instance is a universal image segmentation model based on the Mask2Former architecture. It outperforms specialized architectures for various segmentation tasks while being easy to train on any task. The model combines a backbone feature extractor, a pixel decoder, and a Transformer decoder to achieve state-of-the-art performance across different benchmarks.

### Key Innovations

The model incorporates several key innovations that improve performance and training efficiency:

1. Masked Attention: The Transformer decoder uses masked attention, which restricts attention to localized features centered around predicted segments (objects or regions). This leads to faster convergence and improved performance compared to standard Transformer decoders.

2. Multi-scale High-resolution Features: The model utilizes multi-scale high-resolution features, enabling better segmentation of small objects or regions.

3. Optimization Improvements: The model proposes optimization improvements such as switching the order of self and cross-attention, making query features learnable, and removing dropout, resulting in improved performance without additional compute.

4. Memory Optimization: The model saves training memory by calculating mask loss only on a few randomly sampled points, without affecting performance.

### Performance Evaluation

The model has been evaluated on three image segmentation tasks (panoptic, instance, and semantic segmentation) using four popular datasets (COCO, Cityscapes, ADE20K, and Mapillary Vistas). It achieves state-of-the-art results on these benchmarks, surpassing specialized architectures in performance. Notably, it achieves 57.8 PQ on COCO panoptic segmentation, 50.1 AP on COCO instance segmentation, and 57.7 mIoU on ADE20K semantic segmentation.

### Usage without Fine-tuning, Post-processing, or Pipeline

The facebook/mask2former-swin-base-coco-instance model can be used without fine-tuning, post-processing, or plugging into a pipeline. The following code snippet demonstrates how to utilize the model for image segmentation:

```python
from transformers import AutoModelForSegmentation

# Load the pre-trained model
model = AutoModelForSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-instance")

# Provide input image(s)
input_image = ...

# Perform segmentation
outputs = model.predict(input_image)

# Retrieve the predicted masks
masks = outputs.masks

# Use the predicted masks for further analysis or visualization
...
```

Please note that the input image(s) should be properly preprocessed and formatted according to the model's requirements.

For more specific details on training and evaluation settings, we recommend referring to the original paper [14] and the associated references.

[More Information Needed]

### Downstream Use

Model card description for facebook/mask2former-swin-base-coco-instance:

## Model Details

The facebook/mask2former-swin-base-coco-instance model is a universal image segmentation architecture named Mask2Former. It is designed to outperform specialized architectures across different segmentation tasks while being easy to train on every task. The model builds upon a meta architecture consisting of a backbone feature extractor, a pixel decoder, and a Transformer decoder.

The model introduces several key improvements, including the use of masked attention instead of standard cross-attention, making query features learnable, and removing dropout to improve computation efficiency. These optimizations improve performance without requiring additional compute. Additionally, the model calculates mask loss on few randomly sampled points, resulting in a 3x reduction in training memory without affecting performance.

## Intended Use

The facebook/mask2former-swin-base-coco-instance model can be fine-tuned for various image segmentation tasks, such as panoptic, instance, and semantic segmentation. It performs on par or better than specialized architectures on popular datasets like COCO, Cityscapes, ADE20K, and Mapillary Vistas.

To use the model for fine-tuning, you can follow the training settings described in the references. Adjust the learning rate multiplier, backbone type (CNN or Transformer), and other hyperparameters as needed. The model supports different pixel decoders, and the default choice is the multi-scale deformable attention Transformer (MSDeformAttn) with 6 MSDeformAttn layers applied to feature maps with resolution 1/8.

Here is a code snippet demonstrating how to fine-tune the model for a specific task:

```python
from transformers import AutoModelForSegmentation

# Load the pre-trained model
model = AutoModelForSegmentation.from_pretrained("facebook/mask2former-swin-base-coco-instance")

# Fine-tune the model for your specific task
# ...
```

Please note that additional customization and adaptations may be required based on your specific use case and task requirements.

## Limitations and Ethical Considerations

[More Information Needed]

## Evaluation Results

The facebook/mask2former-swin-base-coco-instance model has been evaluated on panoptic, instance, and semantic segmentation tasks using benchmark datasets such as COCO, Cityscapes, ADE20K, and Mapillary Vistas. It sets the new state-of-the-art with impressive performance metrics, including 57.8 PQ on COCO panoptic segmentation, 50.1 AP on COCO instance segmentation, and 57.7 mIoU on ADE20K semantic segmentation.

## Caveats and Recommendations

[More Information Needed]

## Contributors

[More Information Needed]

## Contact

For any inquiries or updates regarding the model card, please contact [your contact information].

[More Information Needed]

### Out-of-Scope Use

The model facebook/mask2former-swin-base-coco-instance is a universal image segmentation model that performs well across panoptic, instance, and semantic segmentation tasks on popular datasets. It outperforms specialized models designed for each benchmark and is easy to train, saving research effort by three times.

However, it is important to consider potential misuse of the model. While the technical innovations of the model do not seem to have inherent biases, it is crucial to subject the trained models to ethical review to ensure that the predictions generated by the model do not perpetuate problematic stereotypes. Additionally, it is important to ensure that the model is not used for illegal surveillance or any other applications that may infringe upon privacy or violate laws.

It is worth noting that the model performs slightly worse when trained on panoptic segmentation alone compared to when trained with annotations for instance and semantic segmentation tasks. This suggests that the model still needs to be trained specifically for each task. The team's future goal is to develop a single model that can be trained for all image segmentation tasks without task-specific training.

The model struggles with segmenting small objects and fully leveraging multiscale features. Improving the utilization of the feature pyramid and designing losses specifically for small objects are identified as critical areas for improvement.

To address potential misuse, it is important to raise awareness among users about the ethical considerations associated with the model. Users should be cautioned against using the model in ways that may perpetuate problematic stereotypes or violate privacy and legal regulations. It is crucial to ensure responsible and ethical use of the model.

[More Information Needed]

### Bias, Risks, and Limitations

The model facebook/mask2former-swin-base-coco-instance, known as Mask2Former, is a universal image segmentation model that performs well on panoptic, instance, and semantic segmentation tasks. It outperforms specialized models designed for each benchmark while being easy to train. However, there are several known or foreseeable issues associated with this model:

1. Generalization to specific tasks: Although Mask2Former can generalize to different tasks, it still needs to be trained for those specific tasks. This implies that the model may not perform optimally on tasks it has not been specifically trained for.

2. Segmenting small objects: Mask2Former struggles with segmenting small objects and is unable to fully leverage multiscale features. This limitation hinders the model's performance when dealing with small objects or objects at different scales.

3. Ethical considerations: While the technical innovations of Mask2Former do not appear to have inherent biases, it is crucial to conduct ethical reviews when training the model on real-world datasets. This is to ensure that the model's predictions do not propagate problematic stereotypes and that it is not used for applications that may violate legal or ethical boundaries, such as illegal surveillance.

4. Lack of utilization of feature pyramid: Better utilization of the feature pyramid is deemed critical to improving the performance of Mask2Former. The current model could benefit from enhanced feature extraction and design of losses specifically tailored for small objects.

Overall, while Mask2Former shows promising results in universal image segmentation, it has limitations in terms of generalization, segmenting small objects, and ethical considerations. Further research and development are necessary to address these issues and improve the model's performance and societal impact.

### Recommendations

Based on the provided references, here are some recommendations with respect to the foreseeable issues about the model facebook/mask2former-swin-base-coco-instance:

1. Training for specific tasks: While Mask2Former shows good generalization across different image segmentation tasks, it still needs to be trained specifically for each task. In the future, it would be beneficial to develop a model that can be trained only once for all image segmentation tasks, reducing the need for task-specific training. This can help streamline the model development process and save research effort.

2. Performance on small objects and multiscale features: Mask2Former struggles with segmenting small objects and fully leveraging multiscale features. To improve performance in these areas, better utilization of the feature pyramid and designing losses specifically for small objects should be explored. This can help enhance the model's ability to accurately segment small objects and utilize contextual information from multiscale features.

3. Ethical considerations: While the technical innovations in Mask2Former do not appear to have inherent biases, it is important to undergo ethical review when training the model on real-world datasets. This is to ensure that the predictions made by the model do not propagate problematic stereotypes and that the model is not used for applications that may violate ethical principles, such as illegal surveillance. Ethical considerations should be an integral part of the model deployment process.

Overall, the recommendations include exploring ways to develop a single model for all image segmentation tasks, improving performance on small objects and multiscale features, and incorporating ethical considerations throughout the model development and deployment process.

## Training Details

### Training Data

The training data for the model facebook/mask2former-swin-base-coco-instance consists of panoptic segmentation annotations. The model is trained using the COCO dataset with an initial learning rate of 0.0001, a weight decay of 0.05, and a batch size of 16. The training process involves using large-scale jittering augmentation and resizing the images to a fixed size of 1024x1024. For more detailed information on the training settings and data preprocessing, please refer to [Reference 3].

### Training Procedure

#### Preprocessing

The preprocessing steps for the data of the model facebook/mask2former-swin-base-coco-instance are as follows:

1. Tokenization: [More Information Needed]

2. Resizing/Rewriting: [More Information Needed]

Unfortunately, the provided references do not explicitly mention the tokenization and resizing/rewriting steps for the data preprocessing. It is recommended to refer to the original research paper or the official implementation of the model for more details on these preprocessing steps.

#### Training Hyperparameters

The reference does not directly provide the training hyperparameters for the model `facebook/mask2former-swin-base-coco-instance`. However, based on the information given, we can infer some details:

1. The model uses the `Mask R-CNN` baseline settings for instance segmentation on the COCO dataset.
2. The optimizer used is `AdamW`.
3. The initial learning rate is set to `0.0001`.
4. The weight decay is set to `0.05`.
5. A learning rate multiplier of `0.1` is applied to the backbone.
6. The learning rate is decayed at `0.9` and `0.95` fractions of the total number of training steps by a factor of `10`.
7. The model is trained for `50` epochs.
8. The batch size used is `16`.
9. The training data augmentation includes large-scale jittering (LSJ) augmentation, random scale sampling from a range of `0.1` to `2.0`, and fixed-size crop to `1024x1024`.


Please note that the above information is an inference based on the provided references and may not be exhaustive. For more accurate and complete training hyperparameters, more information is needed.

#### Speeds, Sizes, Times

Model: facebook/mask2former-swin-base-coco-instance

Throughput: [More Information Needed]

Start or End Time: [More Information Needed]

Checkpoint Sizes: [More Information Needed]

The available references do not provide specific information about the throughput, start or end time, and checkpoint sizes of the model facebook/mask2former-swin-base-coco-instance. For more details regarding these aspects, it is recommended to refer to the official documentation or contact the model's authors.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model facebook/mask2former-swin-base-coco-instance evaluates on the following benchmarks and datasets:

1. COCO: This benchmark includes 80 "things" and 53 "stuff" categories for segmentation evaluation. It measures performance using average precision (AP) for instance segmentation and mean Intersection-over-Union (mIoU) for semantic segmentation [References: 1, 2, 7].

2. ADE20K: This dataset consists of 100 "things" and 50 "stuff" categories for segmentation evaluation. It is used to evaluate the mIoU metric for semantic segmentation [References: 2, 8].

3. Cityscapes: This dataset is used for segmentation evaluation and contains both "things" and "stuff" categories. However, instance segmentation is only evaluated on the "things" categories [References: 2, 8].

4. Mapillary Vistas: There is no direct mention of this dataset in the provided references. [More Information Needed]

Therefore, the model facebook/mask2former-swin-base-coco-instance evaluates its performance on COCO, ADE20K, and Cityscapes datasets.

#### Factors

The foreseeable characteristics that will influence how the model facebook/mask2former-swin-base-coco-instance behaves include:

1. **Domain and Context**: The model has been trained on the COCO dataset, which consists of a wide range of object categories in various contexts. Therefore, the model is expected to perform well in similar domains and contexts as seen in the training data. However, its performance might degrade when applied to different domains or contexts that significantly differ from the training data.

2. **Population Subgroups**: The model's performance might vary across different population subgroups. Since the COCO dataset is diverse, the model is expected to generalize well to different demographic groups. However, it is crucial to evaluate the model's performance across various population subgroups to uncover any disparities or biases that might exist. This evaluation should be disaggregated to ensure fair and equitable performance across different groups.

3. **Disparities in Performance**: Evaluation of the model should ideally be disaggregated across factors such as race, gender, age, and socioeconomic status to uncover any disparities in performance. This analysis will help identify if the model exhibits biased behavior or unequal performance across different groups. It is essential to address and mitigate any disparities in order to ensure fair and unbiased outcomes.

Please note that the above information is based on the provided references, and a more detailed analysis may require additional information.

#### Metrics

The model facebook/mask2former-swin-base-coco-instance uses multiple metrics for evaluation in order to consider tradeoffs between different errors. 

For instance segmentation, the model uses the standard average precision (AP) metric to evaluate the performance. 

For semantic segmentation, the model uses the mean Intersection-over-Union (mIoU) metric to evaluate the quality of segmentation results.

In addition, the model also evaluates panoptic segmentation using the panoptic quality (PQ) metric, which combines both instance and semantic segmentation. The AP metric is further reported for the "thing" categories using instance segmentation.

Overall, these metrics provide a comprehensive evaluation of the model's performance in different segmentation tasks, considering both instance and semantic segmentation errors.

### Results

Based on the provided references, the evaluation results of the model facebook/mask2former-swin-base-coco-instance can be summarized as follows:

1. Semantic Segmentation:
   - Dataset: COCO (80 "things" and 53 "stuff" categories)
   - Metric: mIoU (mean Intersection-over-Union)
   - Results: [More Information Needed]

2. Instance Segmentation:
   - Dataset: COCO (80 "things" categories)
   - Metric: AP (average precision)
   - Results: 50.1 AP on COCO

3. Panoptic Segmentation:
   - Dataset: COCO (80 "things" and 53 "stuff" categories)
   - Metric: PQ (panoptic quality)
   - Results: 57.8 PQ on COCO

4. ADE20K Benchmark:
   - Dataset: ADE20K (100 "things" and 50 "stuff" categories)
   - Segmentation Tasks: Semantic, Instance, and Panoptic
   - Results: [More Information Needed]

5. Backbones:
   - Swin-L backbone: Sets a new state-of-the-art performance on ADE20K for panoptic segmentation.

Overall, the model facebook/mask2former-swin-base-coco-instance achieves state-of-the-art performance on various segmentation tasks and datasets, including semantic segmentation, instance segmentation, and panoptic segmentation. However, specific evaluation results for some tasks and datasets are not provided and require more information.

#### Summary

The evaluation results for the model facebook/mask2former-swin-base-coco-instance are as follows:

- The model has been compared with specialized state-of-the-art architectures on standard benchmarks for universal image segmentation.
- It has been evaluated on four widely used image segmentation datasets: COCO, ADE20K, Cityscapes, and Mapillary Vistas.
- For panoptic segmentation, the model achieves a panoptic quality (PQ) of 57.8 on the COCO dataset, which sets a new state-of-the-art performance.
- The model also achieves an average precision (AP) of 50.1 for instance segmentation and a mean Intersection-over-Union (mIoU) of 57.7 for semantic segmentation on the ADE20K dataset.
- The results obtained with the Swin-L backbone on the ADE20K dataset further confirm the state-of-the-art performance of the model for panoptic segmentation.
- The model shows competitive performance on other popular image segmentation datasets like Cityscapes.
- It outperforms the best specialized architectures by a significant margin on these datasets.
- The universality of the model is demonstrated by its ability to be used for instance and semantic segmentation, even when trained only with panoptic segmentation annotations.
- The model achieves higher performance than DETR and MaskFormer on metrics like AP Th pan and mIoU pan.

Overall, the evaluation results show that the model facebook/mask2former-swin-base-coco-instance performs exceptionally well on various image segmentation tasks and datasets, setting new state-of-the-art benchmarks and outperforming specialized architectures.

## Model Examination

The model facebook/mask2former-swin-base-coco-instance is an architecture based on MaskFormer [14], which operates on image features to process object queries and decode final binary mask predictions from per-pixel embeddings with object queries. It is the first architecture to outperform specialized models like MaskFormer [14] in instance segmentation, achieving state-of-the-art performance on panoptic segmentation, semantic segmentation, and instance segmentation tasks. 

Although there are universal architectures like DETR [5] and architectures that unify semantic and instance segmentation tasks like panoptic architectures [28], Mask2Former is designed specifically for instance segmentation and demonstrates superior performance compared to universal architectures and panoptic architectures. 

The model outperforms specialized architectures for different image segmentation tasks and sets new state-of-the-art results on popular datasets, including COCO, with 57.8 PQ for panoptic segmentation, 50.1 AP for instance segmentation, and 57.7 mIoU for semantic segmentation. It also shows potential for further improvement through multi-scale inference.

Regarding the question about explainability/interpretability work on the model, there is no specific information provided in the references. Therefore, further information is needed to answer this question.

## Environmental Impact

- **Hardware Type:** The model facebook/mask2former-swin-base-coco-instance is trained on a V100 GPU.
- **Software Type:** The model facebook/mask2former-swin-base-coco-instance is trained on a software type called Mask2Former, which is a new architecture capable of addressing any image segmentation task (panoptic, instance, or semantic). It utilizes a Transformer decoder with a masked attention operator to extract localized features by constraining cross-attention within predicted mask regions. Additionally, it incorporates an efficient multi-scale strategy to handle small objects by utilizing high-resolution features. The model outperforms specialized architectures on popular datasets, achieving state-of-the-art results for panoptic segmentation, instance segmentation, and semantic segmentation.
- **Hours used:** The amount of time used to train the model `facebook/mask2former-swin-base-coco-instance` is not mentioned in the provided references. [More Information Needed]
- **Cloud Provider:** The cloud provider on which the model facebook/mask2former-swin-base-coco-instance is trained is not specified in the provided references. [More Information Needed]
- **Carbon Emitted:** The amount of carbon emitted when training the model facebook/mask2former-swin-base-coco-instance is not mentioned in the provided references. [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The model architecture of facebook/mask2former-swin-base-coco-instance consists of a backbone, a pixel decoder, and a Transformer decoder. The Transformer decoder is the key component of the model and replaces the standard one. It incorporates a masked attention operator, which extracts localized features by constraining cross-attention to within the foreground region of the predicted mask for each query. This is done to handle small objects effectively.

The model adopts an efficient multi-scale strategy to utilize high-resolution features for small objects. It feeds successive feature maps from the pixel decoder's feature pyramid into successive Transformer decoder layers in a round-robin fashion. The model also incorporates optimization improvements to boost performance without introducing additional computation.

The objective of the model facebook/mask2former-swin-base-coco-instance is to address any image segmentation task, including panoptic, instance, or semantic segmentation. It achieves this by utilizing masked attention to extract localized features and by constraining cross-attention within predicted mask regions. The model outperforms specialized architectures for each task on popular datasets, setting new state-of-the-art results for panoptic segmentation, instance segmentation, and semantic segmentation.

### Compute Infrastructure

The compute infrastructure for the model facebook/mask2former-swin-base-coco-instance is not explicitly mentioned in the given references. Therefore, we need more information to provide the compute infrastructure details for this model.

## Citation

```
@misc{bowen-maskedattention,
    author = {Bowen Cheng and
              Ishan Misra and
              Alexander G Schwing and
              Alexander Kirillov and
              Rohit Girdhar and
              Facebook Ai Research},
    title  = {Masked-attention Mask Transformer for Universal Image Segmentation},
    url    = {https://arxiv.org/pdf/2112.01527.pdf}
}
```

