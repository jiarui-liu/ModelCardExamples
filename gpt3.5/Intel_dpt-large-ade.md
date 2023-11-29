# Model Card for Intel/dpt-large-ade

The Intel/dpt-large-ade model is a dense prediction architecture based on an encoder-decoder design using a transformer as the basic computational building block. It sets a new state of the art for tasks such as semantic segmentation and monocular depth estimation on datasets like ADE20K, Pascal Context, NYUv2, and KITTI.

## Model Details

### Model Description

Model Name: Intel/dpt-large-ade

Description:
The Intel/dpt-large-ade model is a dense vision transformer designed for the task of semantic segmentation. It achieves state-of-the-art performance on the ADE20K and Pascal Context datasets. This model is an extension of the DPT architecture, which combines patch-based embeddings and transformer layers to generate fine-grained predictions. The DPT-Large variant is used in this model, which employs 24 transformer layers and a wider feature size D.

Architecture:
The DPT-Large architecture consists of a patch-based embedding procedure, where non-overlapping square patches are extracted from the input image. The embedding procedure projects the flattened patches to a feature dimension of D = 1024. The tokens are then passed through 24 transformer layers, which transform the tokens into new representations. The DPT-Large architecture reassembles tokens from layers l = {5, 12, 18, 24} to generate an image-like representation at multiple resolutions.

Training Procedures:
The DPT-Large model is trained on the ADE20K semantic segmentation dataset for 240 epochs. The training process involves optimizing the model's parameters using a chosen loss function. Unfortunately, further details about the specific training procedures are not provided in the references.

Parameters:
The model has a total of 355 million parameters. Unfortunately, the breakdown of parameters by layer or component is not provided in the references.

Important Disclaimers:
The references do not mention any specific disclaimers about the Intel/dpt-large-ade model. However, it is important to note that the model's performance may vary depending on the specific dataset and task used for evaluation. Additionally, the references provide limited technical details, so further information may be needed to fully understand the model's capabilities and limitations.

For more information, please refer to the references provided.

- **Developed by:** René Ranftl; Alexey Bochkovskiy; Vladlen Koltun
- **Funded by:** The people or organizations that fund the project of the model Intel/dpt-large-ade are not mentioned in the provided references. [More Information Needed]
- **Shared by:** The contributors who made the model Intel/dpt-large-ade available online as a GitHub repo are the Intel Intelligent Systems Lab (Intel-isl).
- **Model type:** The model Intel/dpt-large-ade is a dense prediction transformer (DPT) architecture trained using a transformer-based encoder-decoder design for semantic segmentation tasks, making it a deep learning model using the modality of image analysis.
- **Language(s):** The model Intel/dpt-large-ade uses and processes natural human language for dense prediction tasks, specifically for semantic segmentation.
- **License:** The model Intel/dpt-large-ade uses the MIT License. You can find the license and more information about it in the following link: [MIT License](https://opensource.org/licenses/MIT).
- **Finetuned from model:** Based on the provided references, I can answer the question about the model Intel/dpt-large-ade. 

The reference does not explicitly mention whether the model Intel/dpt-large-ade is fine-tuned from another model. Therefore, it is unclear if it is fine-tuned from another model or not. [More Information Needed]
### Model Sources

- **Repository:** https://github.com/isl-org/DPT
- **Paper:** https://arxiv.org/pdf/2103.13413.pdf
- **Demo:** The link to the demo of the model Intel/dpt-large-ade is "[More Information Needed]".
## Uses

### Direct Use

To use the model Intel/dpt-large-ade without fine-tuning, post-processing, or plugging into a pipeline, you can follow these steps:

1. Download the model weights and place them in the `weights` folder.

   - [dpt_large-ade20k-b12dca68.pt](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-ade20k-b12dca68.pt), [Mirror](https://drive.google.com/file/d/1foDpUM7CdS8Zl6GPdkrJaAOjskb7hHe-/view?usp=sharing)

2. Place one or more input images in the folder `input`.

3. Run the monocular depth estimation model using the following Python code snippet:

   ```python
   python run_monodepth.py -t dpt_large
   ```

   This will use the `dpt_large-ade20k-b12dca68.pt` model weights to perform depth estimation on the input images.

4. The results will be written to the folder `output_monodepth`.

Please note that if you need more information about the specific code implementation or additional details about the model's usage, you can refer to the provided links in the references section.

### Downstream Use

The Intel/dpt-large-ade model can be used for fine-tuning on specific tasks or integrated into larger ecosystems or applications. 

To fine-tune the model for a task, you can follow the steps mentioned in the references. It is recommended to fine-tune the DPT-Hybrid model on smaller datasets such as the KITTI or NYUv2 datasets. The model should be aligned with the ground truth predictions and trained with the loss proposed by Eigen et al.

To integrate the model into a larger ecosystem or app, you can use the pre-trained weights of the model available in the "dpt_large-ade20k-b12dca68.pt" file. You can download the weights and place them in the "weights" folder. Then, you can load the model using the weights file and use it for tasks such as monocular depth estimation or semantic segmentation.

Here is an example code snippet to load the model using PyTorch:

```python
import torch
from torchvision.transforms import functional as F
from PIL import Image

# Load the model
model = torch.hub.load('intel-isl/DPT', 'dpt_large', pretrained=False)
model.load_state_dict(torch.load('path/to/dpt_large-ade20k-b12dca68.pt'))
model.eval()

# Load and preprocess input image
image = Image.open('path/to/input/image.jpg')
image = F.to_tensor(image).unsqueeze(0)

# Inference
with torch.no_grad():
    prediction = model(image)

# Post-process the prediction
# ...

# Use the prediction in your application
# ...
```

Please note that this code snippet is a general example, and you may need to modify it based on your specific use case.

If you need more information or specific code examples, please let me know.

### Out-of-Scope Use

The Intel/dpt-large-ade model is a dense prediction transformer architecture designed for vision tasks such as monocular depth estimation and semantic segmentation. It has demonstrated improved performance compared to fully-convolutional architectures, particularly on datasets like ADE20K and Pascal Context.

Regarding the foreseeable misuse of the model, it is important to highlight that users should not employ the model for any unethical or harmful purposes. Specifically, they should not use it to engage in activities such as:

1. Privacy invasion: The model should not be used to violate the privacy of individuals by analyzing or manipulating images without their consent.

2. Discriminatory practices: Users should not deploy the model in a manner that perpetuates or amplifies biases, discrimination, or unfair treatment towards individuals or groups based on attributes such as race, gender, or age.

3. Misinformation generation: The model should not be utilized to generate or propagate false or misleading information that can be detrimental to society, such as deepfake images or deceptive visual content.

4. Unauthorized surveillance: Users should not employ the model for unauthorized surveillance, including activities that infringe upon personal privacy or violate legal boundaries.

It is crucial for users to adhere to ethical guidelines, legal regulations, and societal norms when utilizing the Intel/dpt-large-ade model to ensure responsible and beneficial use.

### Bias, Risks, and Limitations

The model Intel/dpt-large-ade is a deep learning model for monocular depth estimation. Based on the provided references, here are the known or foreseeable issues stemming from this model:

1. Technical Limitations: The model's performance on the ADE20K dataset is slightly worse compared to fully-convolutional architectures due to the significantly smaller dataset [10]. This indicates the need for a larger dataset to further improve the model's performance.

2. Misunderstandings: While the model sets a new state-of-the-art on the ADE20K and Pascal Context datasets for semantic segmentation, it is important to note that the improvements are attributed to finer-grained and more globally coherent predictions compared to convolutional networks [9]. However, the model's predictions may still contain errors and should not be considered as ground truth.

3. Sociotechnical Limitations: The use of attention-based models and transformers in image analysis has shown promising results, but the full potential of vision transformers like Intel/dpt-large-ade can only be realized with a sufficient amount of training data [8]. This poses challenges in terms of data collection, especially for niche or specialized domains.

4. Foreseeable Harms: The model's performance may vary across different classes in semantic segmentation tasks [1]. Some classes may show significant improvements in per-class Intersection over Union (IoU), while others may not exhibit a strong pattern of improvement. This could lead to unequal performance and potential biases in certain applications.

5. Technical Limitations: The choice of readout token handling in the Reassemble block affects the model's performance. While ignoring the token yields good performance, adding the token leads to worse performance compared to simply ignoring it [2]. The model uses projection for further experiments, but alternative approaches could be explored to improve performance.

6. Technical Limitations: The decision of where to tap features from the transformer backbone is not clear. While convolutional architectures have natural points for passing features, the constant feature resolution in transformers makes it challenging to determine the optimal feature tapping points [5]. Further research and experimentation are needed to identify the most effective feature tapping strategy for improved performance.

In summary, the known or foreseeable issues with the Intel/dpt-large-ade model include technical limitations related to dataset size and readout token handling, potential misunderstandings regarding the model's predictions, sociotechnical limitations related to the availability of training data, foreseeable harms such as varying performance across classes, and technical limitations in determining optimal feature tapping points.

### Recommendations

Based on the references provided, the recommendations with respect to the foreseeable issues about the model Intel/dpt-large-ade are as follows:

1. Readout token: The model uses projection for handling the readout token, which provides slightly better performance compared to ignoring the token. It is recommended to continue using projection for all further experiments.

2. Backbone architecture: The model uses ViT-Base as the backbone architecture, unless specified otherwise. It is recommended to consider using ViT-Base as the backbone architecture for optimal results.

3. Ablation studies: The model has been tested on a reduced meta-dataset consisting of three datasets with high-quality ground truth. It is recommended to perform ablation studies on a wide range of datasets to validate the model's performance across different scenarios.

4. Hybrid architecture: The model's hybrid architecture incorporates low-level features from the ResNet50 embedding network, which leads to better performance compared to using features solely from the transformer stages. It is recommended to continue using low-level features from the embedding network in experiments involving the hybrid architecture.

5. Skip connections: The model taps features from layers that contain low-level features as well as deeper layers, which has shown to be beneficial. It is recommended to continue tapping features from these layers to maintain performance.

6. Semantic segmentation: The model sets a new state of the art on ADE20K and Pascal Context datasets for semantic segmentation. However, it performs slightly worse on the DPT-Large due to the significantly smaller dataset. It is recommended to explore ways to improve performance on smaller datasets.

7. Fine-tuning on smaller datasets: The model can be fine-tuned on smaller datasets such as NYUv2, KITTI, and Pascal Context, where it also sets the new state of the art. It is recommended to fine-tune the model on these datasets for improved performance.

8. Model availability: The model and its variations are available on GitHub at https://github.com/intel-isl/DPT. It is recommended to make the models easily accessible and continue providing updates and support on the GitHub repository.

[More Information Needed]

## Training Details

### Training Data

The training data for the model Intel/dpt-large-ade is the ADE20K semantic segmentation dataset [54]. Unfortunately, there is no specific information or link provided regarding data pre-processing or additional filtering.

### Training Procedure

#### Preprocessing

The preprocessing steps for the data used in the model Intel/dpt-large-ade are as follows:

1. Tokenization: [More Information Needed]

2. Resizing/Rewriting: The input images are resized to a resolution of 384 pixels. This ensures consistent input size for both semantic segmentation and monocular depth estimation tasks.

3. [More Information Needed]

Unfortunately, the provided references do not provide detailed information on the tokenization and other preprocessing steps. It is recommended to refer to the official documentation or the code implementation for more information on the specific preprocessing steps used in the Intel/dpt-large-ade model.

#### Training Hyperparameters

The training hyperparameters for the model Intel/dpt-large-ade are as follows:

- Training dataset: ADE20K semantic segmentation dataset
- Number of epochs: 240 epochs
- Batch size: 16
- Input image size: Resized such that the longer side is 384 pixels
- Random square crops of size 384 are used for training
- Loss function: Not specified
- Optimizer: Not specified

Please note that more information is needed to provide a complete answer on the loss function and optimizer used for training.

#### Speeds, Sizes, Times

The Intel/dpt-large-ade model is a deep learning model that performs monocular depth estimation and semantic segmentation tasks. It is based on a hybrid encoder architecture, which is a preactivation ResNet50 model with group norm and weight standardization. The encoder consists of four stages that downsample the representation before applying multiple ResNet blocks. Skip connections are tapped after the first (R0) and second stage (R1) to improve performance.

For monocular depth estimation, the model uses residual convolutional units and bilinear interpolation for upsampling the representation. The output head for monocular depth estimation includes an initial convolution that halves the feature dimensions, a second convolution with an output dimension of 32, and a final linear layer.

For semantic segmentation, the model utilizes batch normalization, but it is disabled for monocular depth estimation. The output head for semantic segmentation includes a first convolutional block that preserves the feature dimension and a final linear layer that projects the representation to the number of output classes. Dropout is applied with a rate of 0.1, and bilinear interpolation is used for the final upsampling.

The model achieves strong performance in monocular depth estimation and semantic segmentation tasks, setting a new state of the art on the ADE20K and Pascal Context datasets. It produces finer-grained and more globally coherent predictions compared to convolutional networks.

Unfortunately, the provided references do not contain information about the detail throughput, start or end time, checkpoint sizes, or other specific details about the Intel/dpt-large-ade model. [More Information Needed]

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model Intel/dpt-large-ade evaluates on the ADE20K dataset for semantic segmentation. It also fine-tunes on the Pascal Context dataset. Additionally, the model is fine-tuned on the KITTI and NYUv2 datasets to compare its representational power to existing work.

#### Factors

The foreseeable characteristics that will influence how the model Intel/dpt-large-ade behaves include the domain and context in which it is applied, as well as the population subgroups. The model has been extensively evaluated on tasks such as monocular depth estimation and semantic segmentation, specifically on the ADE20K and Pascal Context datasets. It has shown strong performance in these tasks, setting a new state of the art in semantics segmentation on the ADE20K and Pascal Context datasets. The model's performance has been evaluated using metrics such as accuracy, per-class IoU scores, and visual comparisons. However, the evaluation results do not provide a strong pattern across classes, indicating that the model's performance may vary across different object categories. Therefore, a disaggregated evaluation across factors such as object categories and population subgroups would be ideal to uncover any potential disparities in performance. Overall, the model's performance is influenced by factors such as the specific task, dataset size, and the presence of cleaner and finer-grained delineations of object boundaries compared to convolutional networks.

#### Metrics

The metrics used for evaluation in light of tradeoffs between different errors for the model Intel/dpt-large-ade are not explicitly mentioned in the provided references. Therefore, [More Information Needed].

### Results

Based on the provided references, the evaluation results of the model Intel/dpt-large-ade on the Factors and Metrics are as follows:

Factors:
- Monocular Depth Estimation
- Semantic Segmentation

Metrics:
- Accuracy improvement compared to convolutional networks with similar capacity
- Performance on the ADE20K and Pascal Context datasets
- Finer-grained and more globally coherent predictions
- Per-class IoU scores on the ADE20K validation set

Evaluation results for the model Intel/dpt-large-ade are not explicitly mentioned in the provided references. Therefore, more information is needed to provide specific evaluation results for this model.

#### Summary

The evaluation results of the model Intel/dpt-large-ade show that it significantly improves accuracy for monocular depth estimation and semantic segmentation tasks compared to convolutional networks with a similar capacity. The model sets a new state of the art on the ADE20K and Pascal Context datasets for semantic segmentation. However, the performance of DPT-Large is slightly worse than DPT-Hybrid, likely due to the smaller dataset used for training. The model produces cleaner and finer-grained delineations of object boundaries and shows better global depth arrangement compared to fully-convolutional baselines. Unfortunately, specific numerical evaluation metrics or scores for DPT-Large on the ADE20K dataset are not provided. [More Information Needed]

## Model Examination

The model Intel/dpt-large-ade is a deep learning model developed for monocular depth estimation and semantic segmentation tasks. It is based on a hybrid encoder architecture that combines a preactivation ResNet50 with group norm and weight standardization. The hybrid encoder defines four stages that downsample the representation before applying multiple ResNet blocks. The model also utilizes residual convolutional units.

For monocular depth estimation, the model predicts the inverse depth for each pixel and uses bilinear interpolation for upsampling. It achieves improved performance on datasets with dense, high-resolution evaluations, leading to more fine-grained predictions. The model's performance on semantic segmentation tasks, specifically on the ADE20K and Pascal Context datasets, sets a new state of the art. It produces finer-grained and globally coherent predictions compared to convolutional networks.

The model's attention maps and visualization of reference tokens across encoder layers demonstrate its ability to capture intricate details and global depth arrangement. However, further details and experiments on model explainability/interpretability are not provided in the given references.

For more comprehensive information, additional technical details and qualitative/quantitative results can be found in the provided references.

## Environmental Impact

- **Hardware Type:** The model card description for the model Intel/dpt-large-ade is as follows:

---

Model: Intel/dpt-large-ade

Description: The model Intel/dpt-large-ade is a deep learning model trained for monocular depth estimation and semantic segmentation tasks. It is based on a hybrid encoder architecture that combines a preactivation ResNet50 with group norm and weight standardization. The encoder consists of four stages, each downsampling the representation before applying multiple ResNet blocks. Skip connections are utilized after the first and second stages. The decoder utilizes residual convolutional units for both tasks. Batch normalization is used for semantic segmentation but disabled for monocular depth estimation. The model achieves state-of-the-art performance on the ADE20K and Pascal Context datasets for semantic segmentation.

Hardware: [More Information Needed]

---

To answer the question about the hardware type used for training the model Intel/dpt-large-ade, further information is needed from the provided references.
- **Software Type:** The model Intel/dpt-large-ade is trained using the DPT (Dense Prediction Transformer) architecture. It is fine-tuned on the ADE20K semantic segmentation dataset and sets a new state of the art on this challenging dataset. The DPT architecture is a set-to-set transformer encoder that can handle varying image sizes and incorporates a readout token to capture and distribute global information. The software type used to train this model is not mentioned in the provided references. [More Information Needed]
- **Hours used:** The amount of time used to train the model Intel/dpt-large-ade is not specified in the given references. [More Information Needed]
- **Cloud Provider:** Based on the given references, it is not mentioned which cloud provider the model Intel/dpt-large-ade is trained on. [More Information Needed]
- **Carbon Emitted:** The amount of carbon emitted when training the model Intel/dpt-large-ade is not provided in the given references. [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The model architecture of Intel/dpt-large-ade is based on the Dense Prediction Transformer (DPT) framework. DPT is an encoder-decoder architecture that utilizes the Vision Transformer (ViT) as the backbone. The ViT backbone is responsible for extracting features from the input image. The DPT architecture reassembles the bag-of-words representation provided by ViT into image-like feature representations at multiple resolutions.

The DPT architecture consists of three variants: DPT-Base, DPT-Large, and DPT-Hybrid. DPT-Base uses projection as the readout operation and produces feature maps with 256 dimensions. DPT-Large uses layers l = {5, 12, 18, 24} and produces feature maps with 1024 dimensions. DPT-Hybrid employs a ResNet50 to extract features at 1/16th of the input resolution.

The objective of Intel/dpt-large-ade is dense prediction, specifically semantic segmentation. The model is trained on the ADE20K semantic segmentation dataset for 240 epochs. It sets a new state of the art on the ADE20K and Pascal Context datasets, producing finer-grained and more globally coherent predictions compared to convolutional networks.

Overall, Intel/dpt-large-ade is a deep learning model based on the DPT architecture that achieves state-of-the-art performance in semantic segmentation tasks.

### Compute Infrastructure

The compute infrastructure for the model Intel/dpt-large-ade is not explicitly mentioned in the provided references. Therefore, we need more information to determine the compute infrastructure for this model.

## Citation

```
@misc{ren-vision,
    author = {René Ranftl and
              Alexey Bochkovskiy and
              Vladlen Koltun},
    title  = {Vision Transformers for Dense Prediction},
    url    = {https://arxiv.org/pdf/2103.13413.pdf}
}
```

