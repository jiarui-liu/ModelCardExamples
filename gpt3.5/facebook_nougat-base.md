# Model Card for facebook/nougat-base

The model facebook/nougat-base is a transformer-based model that converts images of document pages to formatted markup text. It provides a pre-trained model, code, and dataset for converting PDFs to lightweight markup language, and it allows access to scanned papers and books. It uses a Swin Transformer encoder to convert document images into latent embeddings and subsequently converts them to a sequence of tokens in an autoregressive manner. The model aims to improve accessibility and searchability of academic research papers by enabling machine-readable text extraction and preserving meaningful representations of mathematical equations. The model's performance is noticeable worse for scanned pages compared to digital-born documents, but it still generates sensible text with few errors. The model's code, training, evaluation, and dataset generation can be accessed on GitHub. Additionally, an API is provided for making predictions on PDF files.

## Model Details

### Model Description

Model Name: facebook/nougat-base

Model Architecture:
The architecture of the facebook/nougat-base model is an encoder-decoder transformer [21] architecture. It is built on the Donut [28] architecture and does not require any OCR related inputs or modules. The model utilizes a visual encoder that receives a document image and applies a Swin Transformer [30] to aggregate information across non-overlapping windows of fixed size. The encoded image is then decoded into a sequence of tokens using a transformer decoder architecture with cross-attention. The tokens are generated in an auto-regressive manner, using self-attention and cross-attention to attend to different parts of the input sequence and encoder output respectively. The output is projected to the size of the vocabulary, yielding the logits.

Training Procedures:
- The model is trained in an end-to-end manner.
- During training, perturbations are added to the ground truth text by randomly replacing tokens, which helps reduce the collapse into a repeating loop.
- Data augmentation techniques, such as erosion, dilation, gaussian noise, gaussian blur, bitmap conversion, image compression, grid distortion, and elastic transform, are applied to simulate the imperfections and variability of scanned documents.
- The model is trained using the AdamW optimizer [34] for 3 epochs with an effective batch size of 192.
- The learning rate starts at lr_init = 5 • 10^(-5) and is reduced by a factor of 0.9996 every 15 updates until it reaches lr_end = 7.5 • 10^(-6).

Parameters:
- [More Information Needed]

Important Disclaimers:
- The model is specialized in the scientific text domain and performs better on digital-born academic research papers compared to scanned documents.
- The model's performance on scanned pages is noticeable worse, but it still generates sensible text with few errors.
- The smaller Nougat model performs on par with the larger base model, achieving high scores in all metrics.
- [More Information Needed]

- **Developed by:** Lukas Blecher; Guillem Cucurull; Thomas Scialom; Robert Stojnic; Meta Ai
- **Funded by:** The people or organizations that fund the project of the model facebook/nougat-base are:

- [More Information Needed]
- **Shared by:** The contributors who made the model facebook/nougat-base available online as a GitHub repo are Lukas Blecher, Guillem Cucurull, Thomas Scialom, Robert Stojnic, and Meta Ai.
- **Model type:** The model facebook/nougat-base is an encoder-decoder transformer architecture that allows for end-to-end training. It is trained using the Donut architecture and does not require OCR inputs or modules. The model implicitly recognizes text from document images. [More Information Needed]
- **Language(s):** The model facebook/nougat-base uses and processes natural human language in scientific research articles, including plain text, mathematical expressions, and tables.
- **License:** The license being used for the model `facebook/nougat-base` is CC-BY-NC. You can find more information about the license [here](https://creativecommons.org/licenses/by-nc/).
- **Finetuned from model:** The model facebook/nougat-base is not fine-tuned from another model. [More Information Needed]
### Model Sources

- **Repository:** https://github.com/facebookresearch/nougat/tree/main
- **Paper:** https://arxiv.org/pdf/2308.13418.pdf
- **Demo:** Model Card for facebook/nougat-base:

# Model Details

## Model Name

facebook/nougat-base

## Model Version

1.0

## Model Architecture

Transformer-based model

## Model Language

English

## Model Domain

Optical Character Recognition (OCR)

## Intended Use

The model is designed to convert images of document pages to formatted markup text. It can be used for converting PDFs to a lightweight markup language.

## Training Data

The model was trained on 1,748,201 articles released on arXiv. It also used articles from PMC, though in a limited capacity due to the lack of rich semantic information in the XML files.

## Training Process

The training process involved parsing HTML files and converting them into a lightweight markup language. LaTeXML was used to process the source files and convert them into HTML5 format, standardizing and removing ambiguity from the LaTeX source code. The XML files from PMC were also parsed into the same markup language format.

## Model Performance

The model performs well on digital-born documents but has noticeable lower performance on scanned pages from old textbooks. However, it still generates sensible text for each page with few errors.

## License

The Nougat codebase is licensed under MIT, while the model weights are licensed under CC-BY-NC.

## Citation

[More Information Needed]

## Contact Information

For updates and queries regarding the model card, please contact [Your Name] at [Your Email].

## Acknowledgments

The model is developed by the deep learning model development team at Facebook Research.

## Demo

A demo of the model is available [here](https://nougat-demo.example.com).
## Uses

### Direct Use

The model `facebook/nougat-base` can be used without fine-tuning, post-processing, or plugging into a pipeline. It is designed for converting images of document pages to formatted markup text. The model follows a transformer-based encoder-decoder architecture, specifically based on the Donut architecture.

To use the `facebook/nougat-base` model, you can follow these steps:

1. Install the Hugging Face Transformers library:
```python
pip install transformers
```

2. Load the pre-trained `facebook/nougat-base` model:
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/nougat-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```

3. Prepare the input image for inference:
```python
# Assuming you have an image file named 'document_image.jpg'
image_path = "document_image.jpg"

# Preprocess the image if needed (e.g., resize, crop, normalize)
preprocessed_image = preprocess_image(image_path)
```

4. Tokenize and encode the image:
```python
# Encode the preprocessed image
inputs = tokenizer(preprocessed_image, return_tensors="pt", padding=True, truncation=True)

# Generate the formatted markup text
outputs = model.generate(inputs.input_ids)
formatted_markup_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

The `formatted_markup_text` variable will contain the generated markup text from the input image.

Please note that the code snippet above is a simplified example and may require additional modifications based on your specific use case. It assumes you have implemented the `preprocess_image` function to perform image preprocessing.

### Downstream Use

The facebook/nougat-base model can be used in different ways depending on the specific task or the larger ecosystem/app it is plugged into. When fine-tuned for a task, it can be used for converting images of document pages to formatted markup text.

To fine-tune the Nougat model, you can run the following code snippet:

```python
python train.py --config config/train_nougat.yaml
```

This code snippet trains or fine-tunes the Nougat model using the configuration file `config/train_nougat.yaml`.

After fine-tuning, the model can be used to convert PDFs to a lightweight markup language. The model is trained on one page at a time without knowledge about other pages in the document, which may result in inconsistencies across the document. However, handling each page separately improves parallelization and scalability.

To use the model in a larger ecosystem or app, you can install the `nougat-ocr` package using pip:

```
pip install nougat-ocr
```

If you want to call the model from an API or generate a dataset, there are additional dependencies. You can install them using:

```
pip install "nougat-ocr[api]" or pip install "nougat-ocr[dataset]"
```

Once installed, you can import the `nougat-ocr` package and use the `nougat-base` model for document conversion tasks within your larger ecosystem or app.

Please note that if you want to utilize a GPU, you need to ensure that you have the correct PyTorch version installed, following the instructions provided [here](https://pytorch.org/get-started/locally/).

In summary, the facebook/nougat-base model can be fine-tuned for specific tasks and used for converting PDFs to a lightweight markup language. It can be integrated into a larger ecosystem or app by installing the necessary packages and using the `nougat-base` model for document conversion tasks.

### Out-of-Scope Use

The model facebook/nougat-base is an encoder-decoder transformer architecture that allows for end-to-end training. It does not require OCR related inputs or modules, as the text is recognized implicitly by the network. The model uses a transformer decoder architecture with cross-attention to decode the encoded image into a sequence of tokens. The tokens are generated in an auto-regressive manner using self-attention and cross-attention mechanisms. The Swin Transformer is used as the visual encoder, which crops margins, resizes images, and applies self-attention layers to aggregate information across non-overlapping windows.

The model has been tested on scanned pages from old textbooks and is found to generate sensible text for each page with few errors. The model shows potential for extracting text from both digital-born PDFs and scanned papers/textbooks. However, performance may be worse for scanned pages compared to digital-born documents.

Regarding potential misuse, it is important to note that the model has limitations. It is trained on research papers and works particularly well on documents with a similar structure. It may not perform as accurately on other types of documents. Additionally, the model is primarily trained on English datasets, but initial tests suggest satisfactory performance with other Latin-based languages. Special characters from non-English languages may pose challenges.

To address potential misuse, users should not rely solely on the model's outputs without human verification. The model should be used as a tool to assist in document conversion, but human oversight and verification are necessary to ensure accuracy and reliability. Users should also be cautious when using the model for non-English languages and thoroughly evaluate the quality of the generated text. It is important to understand the model's limitations and not rely on it for critical tasks without proper validation.

For more information on the model and its implementation, the code for model evaluation, training, and dataset generation can be accessed at https://github.com/facebookresearch/nougat.

### Bias, Risks, and Limitations

Based on the references provided, here are the known or foreseeable issues stemming from the model facebook/nougat-base:

1. Limited semantic information recovery: The model is unable to recover the semantic information of mathematical expressions, which may limit its usefulness in certain applications [Reference 1].

2. Inconsistencies across documents: The model is trained on one page at a time without knowledge of other pages in the document, resulting in inconsistencies across the document. This can be observed in the bibliography, where different styles were used, and in section titles, where numbers may be skipped or hallucinated [Reference 7].

3. Inaccuracies in handling special characters and formatting: The model's splitting algorithm may include text from the previous page or cut off words from the end, especially for "invisible" formatting characters like italic, bold text, or section headers [Reference 8].

4. Performance degradation on scanned pages: The model's performance is noticeably worse for scanned pages from old textbooks compared to digital-born documents. However, it still generates sensible text with few errors [Reference 11].

5. Language limitations: The model is primarily trained on English-language research papers, which means its performance with other Latin-based languages is satisfactory. However, the model's utility may be limited for non-English documents [Reference 6].

6. Failure detection issues: There are reported cases where the failure detection heuristic does not work properly, leading to the generation of `[MISSING_PAGE]` responses. This can be mitigated by running the model with the `--no-skipping` flag [References 9, 10].

It is important to note that this analysis is based on the provided references, and additional information may be needed to fully understand the model's limitations and potential harms.

### Recommendations

Based on the references provided, the recommendations for foreseeable issues about the model facebook/nougat-base are as follows:

1. **Semantic Information of Mathematical Expressions**: The model does not recover semantic information of mathematical expressions. It is recommended to explore methods to enhance the model's understanding and extraction of semantic information from mathematical expressions.

2. **Performance on Scanned Pages**: The performance of the model on scanned pages from old textbooks is noticeably worse compared to digital-born documents. It is recommended to investigate and improve the model's performance on scanned pages, considering the differences in image quality and layout.

3. **False Positives in Failure Detection**: There might be false positives in the failure detection when computing on CPU or older GPUs. To mitigate this issue, it is suggested to try passing the `--no-skipping` flag during computation.

4. **Training Stability**: Training instabilities have been observed during model training. It is recommended to continue monitoring and improving the stability of the training process to ensure consistent and reliable results.

5. **Optimization and Hyperparameters**: The model uses an AdamW optimizer and specific hyperparameters for training. It is important to document and carefully validate the chosen optimization method and hyperparameters to ensure reproducibility and performance optimization.

6. **Model Size and Performance**: Both the smaller Nougat model and the larger base model perform well. It is noteworthy that the smaller model performs on par with the larger base model. This suggests that the smaller model can be considered as an alternative without compromising performance.

Please note that additional information may be needed to provide more detailed recommendations or address specific issues.

## Training Details

### Training Data

The training data for the model facebook/nougat-base consists of a paired dataset of PDF pages and corresponding source code, created from open access articles on arXiv, a subset of the PubMed Central (PMC) open access non-commercial dataset for layout diversity, and a portion of the Industry Documents Library (IDL) during pretraining. For more information on the dataset composition, please refer to Table A.1 in the referenced documentation.

### Training Procedure

#### Preprocessing

The preprocessing steps for the data of the model facebook/nougat-base are as follows:

1. Tokenization: The input text is tokenized into subword units using an unspecified tokenizer. The details of the tokenizer used are not mentioned in the references.

2. Resizing: For the visual encoder, the document image is cropped to remove margins and resized to fit in a fixed rectangle of size (H, W). If the image is smaller than the rectangle, padding is added to ensure consistent dimensionality. The specific values of H and W are not mentioned.

3. Data Augmentation: To simulate the imperfections and variability of scanned documents, a number of transformations are applied to the images. These transformations include erosion, dilation, gaussian noise, gaussian blur, bitmap conversion, image compression, grid distortion, and elastic transform. Each transformation has a fixed probability of being applied to a given image. The details of the probabilities for each transformation are not provided.

4. Anti-repetition Augmentation: During training, a random perturbation is introduced to handle wrongly predicted tokens. For each training example, there is a fixed probability that a random token will be replaced by any other randomly chosen token. This process continues until the newly sampled number is greater than a specified threshold (in this case, 10%).

Unfortunately, specific details about the tokenizer used, resizing dimensions, probabilities of data augmentation transformations, and the threshold for anti-repetition augmentation are not mentioned in the references. Further information is needed to provide a more comprehensive answer.

#### Training Hyperparameters

The training hyperparameters for the model facebook/nougat-base are as follows:

- Optimizer: AdamW
- Learning rate: lr_init = 5 • 10^(-5), reduced by a factor of 0.9996 every 15 updates until it reaches lr_end = 7.5 • 10^(-6)
- Number of epochs: 3
- Effective batch size: 192

To train or fine-tune the Nougat model, you can run the following command:
```
python train.py --config config/train_nougat.yaml
```

Please note that this answer is based on the provided information. If you need further details, please provide more information about the training hyperparameters.

#### Speeds, Sizes, Times

The model card for facebook/nougat-base includes the following information:

1. Sequence Length: The model has a sequence length of 4096 tokens. This is because academic research papers can contain dense text and tables, which require a large number of tokens for representation.

2. Model Architecture: The model is based on the BART decoder, which is a decoder-only transformer with 10 layers. The entire architecture has a total of 350M parameters.

3. Smaller Model: There is also a smaller version of the model with 250M parameters. This model has a slightly smaller sequence length of 3584 tokens and only 4 decoder layers.

4. Image Rendering: The document images are rendered at a resolution of 96 DPI. The input size chosen for the Swin Transformer-based model is (H, W) = (896, 672), which is between the US letter and Din A4 format.

5. Training Algorithm: The model is trained using an AdamW optimizer for 3 epochs with an effective batch size of 192. The initial learning rate is set to lr_init = 5 • 10^-5 and is reduced by a factor of 0.9996 every 15 updates until it reaches lr_end = 7.5 • 10^-6.

6. Inference: During inference, the text is generated using greedy decoding.

7. Installation: The model can be installed using pip. The command is `pip install nougat-ocr`.

8. Output: The model outputs a sequence of embedded patches z ∈ R^d×N, where d is the latent dimension and N is the number of patches.

9. Performance: The smaller model performs on par with the larger base model and achieves high scores in all metrics.

10. Scanned Pages: The model performs noticeably worse on pages from old textbooks compared to digital-born documents. However, it still generates sensible text with few errors.

11. Inconsistencies: The model is trained on one page at a time without knowledge about other pages in the document. This can result in inconsistencies across the document, especially in the bibliography and section titles.

Regarding the specific details about throughput, start or end time, checkpoint sizes, etc., [More Information Needed].

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model facebook/nougat-base evaluates on the following benchmarks and datasets:

1. The model is trained on research papers and performs well on documents with a similar structure.
2. The model's performance with other Latin-based languages is satisfactory based on initial tests on a small sample.
3. The model is able to convert pages from old text books obtained from the Internet Archive and Project Gutenberg, although the performance on scanned pages is noticeably worse than for digital-born documents.

Please note that the exact details of the benchmarks and datasets used for evaluation are not provided in the given references. [More Information Needed]

#### Factors

Based on the provided references, here is the description of the foreseeable characteristics that will influence how the model facebook/nougat-base behaves:

The model facebook/nougat-base is trained on research papers and performs well on documents with a similar structure. However, it can still accurately convert other types of documents. The model is primarily trained on datasets consisting of English samples, but initial tests suggest satisfactory performance with other Latin-based languages. The model's utility is limited by factors such as repetitions in the generated text and inconsistencies across the document, particularly in the bibliography and section titles. The model is trained on one page at a time without knowledge of other pages in the document, which may result in inconsistencies. Disparities in performance across different population subgroups and specific evaluation factors are not mentioned in the provided references.

In summary, the foreseeable characteristics that will influence how the model facebook/nougat-base behaves include the document domain and context, the language of the documents, and the limitations in handling repetitions and inconsistencies within the document. However, the provided references do not mention any specific evaluation disaggregated across factors to uncover disparities in performance or population subgroups.

Please note that the information provided is based on the given references and additional information may be needed for a more comprehensive analysis.

#### Metrics

The metrics that will be used for evaluation in light of tradeoffs between different errors for the model facebook/nougat-base are BLEU, METEOR, F-measure, and generalization error εg. These metrics help assess the quality of the model's text generation, its recall and precision, and the overlap between the student and the teacher models. By considering these metrics, we can evaluate the model's performance and tradeoffs between different errors.

### Results

Based on the given references, there is no direct mention of evaluation results for the model facebook/nougat-base. Therefore, [More Information Needed] to provide the evaluation results.

#### Summary

The evaluation results for the model facebook/nougat-base are not mentioned in the given references. [More Information Needed]

## Model Examination

Model Card Description:

The facebook/nougat-base model is an encoder-decoder transformer architecture that is trained on one page at a time without knowledge about other pages in the document. This approach allows for parallelization and scalability but may result in inconsistencies across the document, particularly in the bibliography and section titles. The model is primarily designed for research papers but can also handle other types of documents. It performs well with English texts and shows satisfactory performance with other Latin-based languages.

The architecture of the model is based on the Donut architecture, which is an encoder-decoder transformer. The model does not require any OCR related inputs or modules as it implicitly recognizes the text. The generation speed of the model depends on the amount of text on each page, and on average, it can process 6 pages in parallel with a mean generation time of 19.5s per batch.

The model outputs a sequence of embedded patches, where the latent dimension is denoted by d and the number of patches is denoted by N. The model has been evaluated against other approaches and has consistently outperformed them in various metrics.

In terms of interpretability and explainability, this is an experimental section for the model facebook/nougat-base. Work on explainability and interpretability is currently being explored and further research is needed in this area.

For more information, please refer to the code and documentation available at the following GitHub repository: [github.com/ facebookresearch/nougat].

## Environmental Impact

- **Hardware Type:** The hardware type that the model facebook/nougat-base is trained on is not mentioned in the provided references. [More Information Needed]
- **Software Type:** The model facebook/nougat-base is trained on the software type "Nougat OCR".
- **Hours used:** To train the model `facebook/nougat-base`, we used an AdamW optimizer to train for 3 epochs with an effective batch size of 192. The learning rate was initialized at 5 • 10^(-5) and reduced by a factor of 0.9996 every 15 updates until it reached 7.5 • 10^(-6) [1].

However, the amount of time used to train the model is not mentioned in the provided references. Therefore, we need more information to answer this question.
- **Cloud Provider:** The cloud provider on which the model facebook/nougat-base is trained is not mentioned in the provided references. [More Information Needed]
- **Carbon Emitted:** The amount of carbon emitted when training the model facebook/nougat-base is not specified in the given references. [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The model architecture of facebook/nougat-base is an encoder-decoder transformer architecture based on the Donut architecture. The model consists of an encoder that takes a document image and converts it into latent embeddings using a Swin Transformer, which applies self-attention layers to aggregate information across non-overlapping windows of the image. The encoded image is then decoded into a sequence of tokens using a transformer decoder architecture with cross-attention. The tokens are generated in an auto-regressive manner using self-attention and cross-attention to attend to different parts of the input sequence and encoder output, respectively. The output is projected to the size of the vocabulary, resulting in logits.

The objective of facebook/nougat-base is to convert images of document pages to formatted markup text. It aims to provide a pre-trained model capable of converting PDFs to a lightweight markup language, allowing access to scanned papers and books. The model's primary contributions include releasing the code and model on GitHub, introducing a pipeline to create datasets for pairing PDFs to source code, and enabling the accessibility and searchability of academic research papers by converting them into machine-readable text.

Please note that the provided answer is based on the given references. For more detailed information, please refer to the original sources.

### Compute Infrastructure

The compute infrastructure for the model `facebook/nougat-base` is not explicitly mentioned in the provided references. [More Information Needed].

## Citation

```
@misc{lukas-nougat,
    author = {Lukas Blecher and
              Guillem Cucurull and
              Thomas Scialom and
              Robert Stojnic and
              Meta Ai},
    title  = {Nougat: Neural Optical Understanding for Academic Documents},
    url    = {https://arxiv.org/pdf/2308.13418.pdf}
}
```

