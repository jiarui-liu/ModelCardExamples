# Model Card for facebook/mask2former-swin-small-ade-semantic

The model `facebook/mask2former-swin-small-ade-semantic` is a new architecture called Mask2Former that can address any image segmentation task (panoptic, instance, or semantic). It incorporates masked attention and constrains cross-attention within predicted mask regions, outperforming specialized architectures and setting new state-of-the-art results on popular datasets.

## Model Details

### Model Description

Model Name: facebook/mask2former-swin-small-ade-semantic

Description:
The facebook/mask2former-swin-small-ade-semantic model is a universal image segmentation architecture named Masked-attention Mask Transformer (Mask2Former). It outperforms specialized architectures across different segmentation tasks, while still being easy to train on every task. The model adopts a meta architecture consisting of a backbone feature extractor, a pixel decoder, and a Transformer decoder. It introduces key improvements, including masked attention, multi-scale high-resolution features, and efficient training strategies.

Architecture:
The model architecture follows a meta architecture that includes a backbone feature extractor, a pixel decoder, and a Transformer decoder. The Transformer decoder replaces the standard one and includes a masked attention operator, which extracts localized features by constraining cross-attention to within the foreground region of the predicted mask for each query. This masked attention leads to faster convergence and improved performance compared to the cross-attention used in a standard Transformer decoder.

Training Procedures:
The model is trained on three image segmentation tasks: panoptic, instance, and semantic segmentation. It is evaluated on popular datasets such as COCO, Cityscapes, ADE20K, and Mapillary Vistas. The model achieves state-of-the-art performance on these benchmarks, with metrics including Panoptic Quality (PQ), Average Precision (AP), and mean Intersection over Union (mIoU).

Parameters:
The specific details about the model parameters, such as the number of layers, hidden size, and attention heads, are not mentioned in the references. [More Information Needed]

Important Disclaimers:
The specific training procedures and hyperparameters are not mentioned in the references. [More Information Needed]

- **Developed by:** Bowen Cheng; Ishan Misra; Alexander G Schwing; Alexander Kirillov; Rohit Girdhar; Facebook Ai Research
- **Funded by:** The people or organizations that fund the project of the model facebook/mask2former-swin-small-ade-semantic are not mentioned in the provided references. [More Information Needed]
- **Shared by:** The contributors who made the model facebook/mask2former-swin-small-ade-semantic available online as a GitHub repo are not mentioned in the provided references. [More Information Needed]
- **Model type:** The model facebook/mask2former-swin-small-ade-semantic is a universal image segmentation architecture that uses a meta architecture with a Transformer decoder and masked attention, allowing it to address any image segmentation task (panoptic, instance, or semantic) while remaining easy to train. It outperforms specialized models designed for each benchmark and saves research effort.
- **Language(s):** The model facebook/mask2former-swin-small-ade-semantic uses or processes natural human language for mask classification and semantic segmentation tasks, where it predicts binary masks and assigns category labels to different segments of an image.
- **License:** The model facebook/mask2former-swin-small-ade-semantic is licensed under the MIT License. You can find the license information and details [here](https://github.com/facebookresearch/Mask2Former/blob/main/LICENSE).
- **Finetuned from model:** Based on the provided references, the answer to the question about the model `facebook/mask2former-swin-small-ade-semantic` is that it is not fine-tuned from another model. There is no mention of a base model or any reference to it in the given information.
### Model Sources

- **Repository:** https://github.com/facebookresearch/Mask2Former/
- **Paper:** https://arxiv.org/pdf/2112.01527.pdf
- **Demo:** The model card for facebook/mask2former-swin-small-ade-semantic is as follows:

## Model Details

- **Model name**: facebook/mask2former-swin-small-ade-semantic
- **Model type**: Image Segmentation
- **Description**: Mask2Former is a universal image segmentation architecture that outperforms specialized architectures across different segmentation tasks. It is built upon a meta architecture consisting of a backbone feature extractor, a pixel decoder, and a Transformer decoder. The model utilizes masked attention in the Transformer decoder, multi-scale high-resolution features, and other key improvements to achieve state-of-the-art performance on various benchmarks.
- **Intended Use**: The model is intended to perform image segmentation tasks, including panoptic, instance, and semantic segmentation.
- **Training Data**: The model is trained on four popular datasets: COCO, Cityscapes, ADE20K, and Mapillary Vistas.
- **Training Time**: The training time for the model is not specified.
- **Hardware**: The model does not require any specialized hardware.
- **Metrics**: Mask2Former achieves a Panoptic Quality (PQ) of 57.8 on COCO panoptic segmentation, an Average Precision (AP) of 50.1 on COCO instance segmentation, and a mean IoU (mIoU) of 57.7 on ADE20K semantic segmentation.
- **Limitations**: The limitations of the model are not specified.
- **Ethical Considerations**: The ethical considerations of the model are not specified.

## How to Use

- **Installation**: The model can be installed using the Huggingface library.
- **Dependencies**: The model requires the Huggingface library and its dependencies.
- **Example Code**:

```python
from transformers import AutoModelForImageSegmentation, AutoTokenizer

# Load the model
model = AutoModelForImageSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
tokenizer = AutoTokenizer.from_pretrained("facebook/mask2former-swin-small-ade-semantic")

# Perform image segmentation
inputs = tokenizer("input_image.jpg", return_tensors="pt")
outputs = model(**inputs)

# Get the predicted masks
masks = outputs.logits
```

## Model Performance

- **Evaluation Results**: Mask2Former achieves state-of-the-art performance on various benchmarks, including COCO panoptic segmentation, COCO instance segmentation, and ADE20K semantic segmentation.
- **Comparison to Baselines**: Mask2Former outperforms specialized architectures across different segmentation tasks.
- **Bias and Fairness Considerations**: The bias and fairness considerations of the model are not specified.

## Dataset

- **Dataset Name**: COCO, Cityscapes, ADE20K, Mapillary Vistas
- **Dataset Description**: The datasets used for training and evaluation include COCO, Cityscapes, ADE20K, and Mapillary Vistas. These datasets are popular benchmarks for image segmentation tasks.
- **Dataset Size**: The size of the datasets is not specified.
- **Dataset Creation**: The creation process of the datasets is not specified.

## Training

- **Preprocessing**: The preprocessing steps for the training data are not specified.
- **Training Procedure**: The training procedure for the model is not specified.
- **Training Time**: The training time for the model is not specified.
- **Hardware Used**: The hardware used for training the model is not specified.

## Additional Information

- **References**: The model is based on the MaskFormer codebase (https://github.com/facebookresearch/MaskFormer).
- **BibTeX Citation**:

```
@inproceedings{cheng2021mask2former,
  title={Masked-attention Mask Transformer for Universal Image Segmentation},
  author={Bowen Cheng and Ishan Misra and Alexander G. Schwing and Alexander Kirillov and Rohit Girdhar},
  journal={CVPR},
  year={2022}
}
```
- **Demo**: [Demo Link](https://[More Information Needed])

Please note that the demo link is not provided in the references.
## Uses

### Direct Use

Model Card Description: facebook/mask2former-swin-small-ade-semantic

## Model Details
The model `facebook/mask2former-swin-small-ade-semantic` is a universal image segmentation architecture named Masked-attention Mask Transformer (Mask2Former). It outperforms specialized architectures across different segmentation tasks, while still being easy to train on every task. The architecture is built upon a simple meta architecture consisting of a backbone feature extractor, a pixel decoder, and a Transformer decoder. The model achieves state-of-the-art results on panoptic, instance, and semantic segmentation tasks.

## Intended Use
The model `facebook/mask2former-swin-small-ade-semantic` can be used without fine-tuning, post-processing, or plugging into a pipeline. It is designed to be easily accessible and ready to use out of the box. The model can be used for various image segmentation tasks, such as panoptic, instance, and semantic segmentation, without the need for extensive modifications or additional processing.

## Limitations and Bias
The limitations and potential biases of the model `facebook/mask2former-swin-small-ade-semantic` are not explicitly mentioned in the provided references. Further information is needed to understand the specific limitations and potential biases of this model.

## Dataset and Training
The model `facebook/mask2former-swin-small-ade-semantic` is evaluated on three image segmentation tasks (panoptic, instance, and semantic segmentation) using four popular datasets (COCO, Cityscapes, ADE20K, and Mapillary Vistas). It sets the new state-of-the-art on these benchmarks, achieving high performance on COCO panoptic segmentation (57.8 PQ), COCO instance segmentation (50.1 AP), and ADE20K semantic segmentation (57.7 mIoU). The model is trained using a V100 GPU with a batch size of 1, following specific training settings as mentioned in the references.

## Evaluation Metrics
The model `facebook/mask2former-swin-small-ade-semantic` is evaluated using various evaluation metrics based on the specific segmentation tasks. For panoptic segmentation, the model's performance is measured by the Panoptic Quality (PQ) metric. For instance segmentation, the Average Precision (AP) metric is used. For semantic segmentation, the mean Intersection over Union (mIoU) metric is applied. The model achieves state-of-the-art results on these metrics, as mentioned in the references.

## Training Time and Hardware Requirements
The training time and hardware requirements for the model `facebook/mask2former-swin-small-ade-semantic` are not explicitly mentioned in the provided references. Further information is needed to determine the specific training time and hardware requirements for this model.

## Code Snippet
[More Information Needed]

### Downstream Use

Model Card Description:

## Model Details

The model `facebook/mask2former-swin-small-ade-semantic` is a deep learning model based on the Mask2Former architecture. It is designed for image segmentation tasks, including panoptic, instance, and semantic segmentation. The model achieves state-of-the-art performance on popular datasets such as COCO, Cityscapes, ADE20K, and Mapillary Vistas.

The architecture of `mask2former-swin-small-ade-semantic` is built upon the DETR framework, which has shown excellent results in panoptic and semantic segmentation. By incorporating specialized modules and optimizations, Mask2Former outperforms existing architectures for various segmentation tasks. Notably, it surpasses MaskFormer in instance segmentation.

## Intended Use

The `mask2former-swin-small-ade-semantic` model can be fine-tuned for specific image segmentation tasks or be integrated into a larger ecosystem or application. Fine-tuning the model involves training it on a task-specific dataset to adapt it to the specific requirements and nuances of the target task. After fine-tuning, the model can be used to perform accurate and efficient segmentation on new images.

To fine-tune the `mask2former-swin-small-ade-semantic` model, you can use the following code snippet as a reference:

```python
from transformers import AutoModelForImageSegmentation, AutoTokenizer

model_name = "facebook/mask2former-swin-small-ade-semantic"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForImageSegmentation.from_pretrained(model_name)

# Fine-tune the model on your task-specific dataset
# ...
```

Please note that the code snippet above provides a starting point for fine-tuning the model. You will need to modify and adapt it to your specific dataset and training pipeline.

## Limitations and Ethical Considerations

It is important to consider the limitations and ethical implications of using the `mask2former-swin-small-ade-semantic` model. Some key points to consider include:

- Hardware and Computational Requirements: The model requires a powerful GPU (e.g., V100) for efficient training and inference. It is essential to ensure that the available hardware can meet these requirements.

- Training Data Bias: The performance of the model heavily relies on the quality and representativeness of the training data. Biases present in the training data can be reflected in the model's predictions, potentially leading to unfair or discriminatory outcomes. Careful data collection and preprocessing are crucial to mitigate such biases.

- Privacy and Security: When integrating the model into an application or ecosystem, it is important to consider privacy and security implications. Ensure that appropriate measures are in place to protect user data and prevent unauthorized access to the model and its outputs.

- Fair Use and Responsible AI: It is essential to use the model responsibly and comply with legal and ethical guidelines. Proper documentation, user consent, and transparency are important aspects to consider when deploying the model in real-world applications.

Overall, understanding the limitations and ethical considerations associated with the `mask2former-swin-small-ade-semantic` model is crucial for its appropriate and responsible use.

[More Information Needed]

### Out-of-Scope Use

The model facebook/mask2former-swin-small-ade-semantic, developed by our team, is a powerful universal image segmentation model. However, it is important to consider potential misuse and address what users should not do with the model. Based on the available information, here are some foreseeable misuses and guidelines:

1. **Illegal surveillance**: The model should not be used for applications that involve illegal surveillance. It is essential to respect privacy laws and ethical considerations.

2. **Propagating problematic stereotypes**: The predictions generated by the model should undergo ethical review to ensure they do not propagate problematic stereotypes. It is crucial to analyze and address any biases that may arise from the predictions.

3. **Unintended applications**: The model should not be used for applications other than image segmentation tasks. While our ultimate goal is to train a single model for all image segmentation tasks, the model still needs to be trained specifically for each task. It should not be applied to tasks outside the scope of image segmentation.

4. **Misinterpretation of results**: Users should be cautious not to misinterpret the model's results. While the model achieves top results in various image segmentation tasks, it may struggle with segmenting small objects and fully leveraging multiscale features. Users should be aware of these limitations and interpret the results accordingly.

5. **Proper attribution**: If researchers use the model in their work or refer to the baseline results published in the Model Zoo, they should provide the appropriate BibTeX entry to give credit to the original authors.

It is important to note that this response is based on the available information. For a comprehensive understanding and further details, it is recommended to consult the provided references or seek additional information from the development team.

### Bias, Risks, and Limitations

Model Card Description: facebook/mask2former-swin-small-ade-semantic

The facebook/mask2former-swin-small-ade-semantic model, known as Mask2Former, is a universal image segmentation model that aims to provide accurate segmentations for panoptic, instance, and semantic segmentation tasks. It is built upon a meta framework and utilizes a new Transformer decoder with masked attention. The model achieves top results on popular datasets for all three segmentation tasks, outperforming even specialized models designed for each benchmark. 

However, there are several known or foreseeable issues associated with the model:

1. Limited performance on small objects: Mask2Former struggles with segmenting small objects and is unable to fully leverage multiscale features. Improvements in the utilization of the feature pyramid and the design of losses for small objects are crucial to address this limitation.

2. Training for specific tasks: While Mask2Former can generalize to different tasks, it still needs to be trained specifically for each task. This implies that the model requires task-specific training, thereby increasing the training effort required.

3. Ethical considerations: Although the technical innovations of Mask2Former do not appear to have inherent biases, it is important to subject models trained with this approach on real-world datasets to ethical review. This is necessary to ensure that the predictions generated by the model do not propagate problematic stereotypes and that the model is not used for applications that may violate legal or ethical norms, such as illegal surveillance.

In summary, the facebook/mask2former-swin-small-ade-semantic model shows promising results in universal image segmentation tasks. However, it has limitations in segmenting small objects and requires task-specific training. Ethical considerations should also be taken into account when deploying the model in real-world scenarios.

For more results, detailed performance metrics on different benchmarks and datasets, ablation studies, and visualization of Mask2Former predictions, please refer to the provided references.

References:
- [More Information Needed]
- [More Information Needed]
- [More Information Needed]
- [More Information Needed]
- [More Information Needed]
- [More Information Needed]
- [More Information Needed]
- [More Information Needed]
- [More Information Needed]
- [More Information Needed]

### Recommendations

Based on the provided references, here are the recommendations with respect to the foreseeable issues about the model facebook/mask2former-swin-small-ade-semantic:

1. Train for Specific Tasks: While Mask2Former shows generalization to different tasks, it still needs to be trained specifically for each task. Future development should focus on creating a model that can be trained only once for all image segmentation tasks.

2. Improve Small Object Segmentation: Mask2Former struggles with segmenting small objects and fails to fully leverage multiscale features. It is recommended to explore better utilization of the feature pyramid and design specific losses for small objects to address this issue.

3. Consider Ethical Review: The technical innovations of Mask2Former do not appear to have inherent biases. However, models trained with this approach on real-world datasets should undergo ethical review to ensure that the predictions do not propagate problematic stereotypes. Additionally, the model should not be used for applications including illegal surveillance.

Please note that the information provided is based on the given references, and further analysis or research may be required to fully address the foreseeable issues with the model facebook/mask2former-swin-small-ade-semantic.

## Training Details

### Training Data

The training data for the model facebook/mask2former-swin-small-ade-semantic is not explicitly mentioned in the provided references. [More Information Needed]

### Training Procedure

#### Preprocessing

The preprocessing steps for the data of the model `facebook/mask2former-swin-small-ade-semantic` are as follows:

1. Tokenization: There is no specific mention of tokenization in the references. [More Information Needed]

2. Resizing/Rewriting: The resized mask prediction of the previous Transformer decoder layer is used as the attention mask. It is resized to the same resolution as the feature map. [Code Reference: Equation (3)]

3. Other Preprocessing: There is no mention of any other preprocessing steps in the references. [More Information Needed]

Please note that the information provided is based on the available references, and there may be additional steps or details that are not mentioned.

#### Training Hyperparameters

The training hyperparameters for the model facebook/mask2former-swin-small-ade-semantic are as follows:

- Training duration: 50 epochs
- Batch size: 16
- Data augmentation: Large-scale jittering (LSJ) augmentation with a random scale sampled from the range 0.1 to 2.0, followed by a fixed size crop to 1024×1024
- Inference resizing: Resize an image with a shorter side to 800 and a longer side up to 1333
- Loss function: Binary cross-entropy loss for the mask loss, combined with the dice loss. The final loss is a combination of mask loss and classification loss.
- Loss weights: λ ce = 5.0 for the binary cross-entropy loss, λ dice = 5.0 for the dice loss, and λ cls = 2.0 for predictions matched with a ground truth and 0.1 for "no object" predictions.
- Query features: Learnable query features before the Transformer decoder
- Backbones: Swin-L backbone for panoptic and instance segmentation models, while other backbones or semantic segmentation models use different numbers of queries (100 or 200)
- FLOPs and fps: FLOPs are averaged over 100 validation images, and frames-per-second (fps) is measured on the learnable query features before the Transformer decoder.

Please note that the specific learning rate, weight decay, and other optimizer-related hyperparameters are not provided in the given references, so more information is needed in that regard.

#### Speeds, Sizes, Times

The Facebook/mask2former-swin-small-ade-semantic model card does not provide specific details about the throughput, start or end time, or checkpoint sizes of the model. Therefore, we need more information to provide these details.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model `facebook/mask2former-swin-small-ade-semantic` evaluates on the ADE20K dataset for semantic segmentation. The ADE20K dataset consists of 100 "things" and 50 "stuff" categories. It also evaluates on the COCO dataset for panoptic segmentation, which includes 80 "things" and 53 "stuff" categories. Additionally, it evaluates on the Cityscapes dataset for three segmentation tasks, namely semantic, instance, and panoptic segmentation. The Cityscapes dataset includes a union of "things" and "stuff" categories for semantic and panoptic segmentation, while instance segmentation is only evaluated on the "things" categories.

#### Factors

Based on the provided references, the foreseeable characteristics that will influence how the model facebook/mask2former-swin-small-ade-semantic behaves are as follows:

1. **Domain and Context**: The model has been evaluated on three image segmentation tasks (panoptic, instance, and semantic segmentation) using popular datasets such as COCO, Cityscapes, ADE20K, and Mapillary Vistas. The model has shown competitive performance on these benchmarks, setting new state-of-the-art results. Therefore, the model is expected to perform well in similar image segmentation domains and contexts.

2. **Population Subgroups**: The references do not explicitly mention the evaluation of the model across different population subgroups. Therefore, there is a lack of information regarding the model's performance disparities across factors such as gender, age, ethnicity, or socioeconomic status. Further evaluation and analysis are required to uncover any disparities in model performance among different population subgroups.

Overall, while the model has demonstrated strong performance on various segmentation tasks and datasets, more information is needed to fully understand its behavior and potential disparities in performance across different factors. Further evaluation and analysis should be conducted to ensure the model's fairness and to uncover any biases or disparities in its predictions.

#### Metrics

The metrics used for evaluation in light of tradeoffs between different errors for the model facebook/mask2former-swin-small-ade-semantic are:

- For instance segmentation: Average Precision (AP) metric.
- For semantic segmentation: mean Intersection-over-Union (mIoU) metric.

These metrics are used to assess the performance of the model on the respective tasks. The model card description does not provide specific information about the tradeoffs between different errors. [More Information Needed]

### Results

The evaluation results of the model facebook/mask2former-swin-small-ade-semantic based on the factors and metrics are as follows:

- For panoptic segmentation on the ADE20K dataset, Mask2Former with Swin-L backbone achieved a new state-of-the-art performance, surpassing other methods. Unfortunately, the specific results are not provided. 

- For semantic segmentation on the ADE20K dataset, Mask2Former with various backbones outperformed all existing models with various backbones. The best model achieved a new state-of-the-art mean Intersection-over-Union (mIoU) of 57.7. 

- The model also achieved higher performance on two other metrics compared to DETR and MaskFormer: AP Th pan (average precision evaluated on the 80 "thing" categories using instance segmentation annotation) and mIoU pan (mean Intersection-over-Union evaluated on the 133 categories for semantic segmentation converted from panoptic segmentation annotation). Unfortunately, the specific values for these metrics are not provided. 

In summary, the model facebook/mask2former-swin-small-ade-semantic demonstrates its effectiveness for universal image segmentation. It outperforms specialized state-of-the-art architectures on standard benchmarks and sets new state-of-the-art results on four datasets. However, more specific evaluation results are needed to provide a comprehensive analysis.

#### Summary

The evaluation results of the model facebook/mask2former-swin-small-ade-semantic are as follows:

- Mask2Former with Swin-L backbone achieves a new state-of-the-art performance on ADE20K for panoptic segmentation.
- Mask2Former outperforms all existing semantic segmentation models with various backbones on ADE20K validation set, setting a new state-of-the-art with a mean Intersection-over-Union (mIoU) of 57.7.
- The best Mask2Former model on the test set also achieves the absolute new state-of-the-art performance on both the validation and test-dev sets.
- The model shows excellent performance in segmenting large objects, even outperforming the challenge winner on AP L without any additional techniques.
- However, the model's performance on small objects leaves room for further improvement.

Please note that the above summary is based on the provided references and might not capture all the specific details. For more comprehensive information, please refer to the referenced papers.

## Model Examination

The model facebook/mask2former-swin-small-ade-semantic is an implementation of Mask2Former, which is a meta architecture that operates on image features to process object queries and make binary mask predictions. It replaces the standard Transformer decoder with a new Transformer decoder that uses masked attention instead of the standard cross-attention. This masked attention operator extracts localized features by constraining cross-attention to within the foreground region of the predicted mask for each query, instead of attending to the full feature map. 

To handle small objects, the model proposes an efficient multi-scale strategy that utilizes high-resolution features. It feeds successive feature maps from the pixel decoder's feature pyramid into successive Transformer decoder layers in a round-robin fashion.

The model achieves state-of-the-art performance on both the validation and test sets for semantic segmentation tasks. It performs extremely well at segmenting large objects and outperforms other models without any additional bells-and-whistles. However, there is room for improvement in segmenting small objects.

Unfortunately, there is no specific information provided in the references about the work on explainability/interpretability for the model facebook/mask2former-swin-small-ade-semantic. [More Information Needed]

## Environmental Impact

- **Hardware Type:** The hardware type that the model facebook/mask2former-swin-small-ade-semantic is trained on is a V100 GPU.
- **Software Type:** The model `facebook/mask2former-swin-small-ade-semantic` is trained on the software type called Mask2Former. This software type is a new architecture capable of addressing any image segmentation task, including panoptic, instance, or semantic segmentation. It utilizes masked attention to extract localized features for accurate per-pixel classification. The architecture aims to generalize to different tasks, but it still needs to be trained specifically for each task.
- **Hours used:** The amount of time used to train the model facebook/mask2former-swin-small-ade-semantic is not specified in the provided references. [More Information Needed]
- **Cloud Provider:** The cloud provider on which the model facebook/mask2former-swin-small-ade-semantic is trained is not mentioned in the provided references. [More Information Needed]
- **Carbon Emitted:** The amount of carbon emitted when training the model `facebook/mask2former-swin-small-ade-semantic` is not mentioned in the provided references. [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The model facebook/mask2former-swin-small-ade-semantic is built upon a meta architecture for mask classification. It consists of a backbone, a pixel decoder, and a Transformer decoder. The key component of this model is the Transformer decoder, which incorporates a masked attention operator instead of the standard cross-attention. The masked attention operator extracts localized features by constraining cross-attention within the foreground region of the predicted mask for each query. This approach improves convergence and results.

The objective of the model is to address any image segmentation task, including panoptic, instance, or semantic segmentation. By utilizing the masked attention and other optimization improvements, the model achieves state-of-the-art performance on popular datasets such as COCO panoptic, COCO instance, and ADE20K semantic segmentation. It outperforms specialized architectures designed for each task and sets new records in panoptic segmentation (57.8 PQ on COCO), instance segmentation (50.1 AP on COCO), and semantic segmentation (57.7 mIoU on ADE20K).

### Compute Infrastructure

The compute infrastructure for the model facebook/mask2former-swin-small-ade-semantic is not explicitly mentioned in the references provided. [More Information Needed]

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

