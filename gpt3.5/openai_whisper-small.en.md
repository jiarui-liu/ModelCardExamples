# Model Card for openai/whisper-small.en

The model openai/whisper-small.en is a general-purpose speech recognition model that is trained on a large dataset of diverse audio and can perform multilingual speech recognition, speech translation, and language identification.

## Model Details

### Model Description

Model Name: openai/whisper-small.en

Description:
The openai/whisper-small.en model is a Transformer sequence-to-sequence model trained on various speech processing tasks. It is designed to replace many stages of a traditional speech-processing pipeline by jointly representing these tasks as a sequence of tokens to be predicted by the decoder. The model architecture is based on an encoder-decoder Transformer, which has been validated to scale reliably. The model takes audio as input, which is re-sampled to 16,000 Hz and transformed into an 80-channel log-magnitude Mel spectrogram representation.

Training Procedures:
The model is trained using a multitask training format, where different speech processing tasks are jointly represented as tokens to be predicted by the decoder. The training data is globally scaled to be between -1 and 1 with approximately zero mean across the pre-training dataset. The model uses pre-activation residual blocks and applies layer normalization to the encoder output. The decoder uses learned position embeddings and tied input-output token representations. The encoder and decoder have the same width and number of transformer blocks.

Parameters:
The specific number of layers, width, and heads for the openai/whisper-small.en model are not mentioned in the provided references. [More Information Needed]

Important Disclaimers:
1. The openai/whisper-small.en model is built to study the capabilities of large-scale supervised pre-training for speech recognition. It uses an off-the-shelf architecture to avoid confounding the findings with model improvements.
2. The model may have a tendency to transcribe plausible but almost always incorrect guesses for the names of speakers. This is because many transcripts in the pre-training dataset include the name of the person who is speaking, encouraging the model to predict them, but this information is rarely inferable from only the most recent context.
3. The model has been trained on various tasks and datasets, but the specific datasets used for training are not mentioned in the provided references. [More Information Needed]
4. The model has been trained using data parallelism across accelerators and with FP16 precision. It uses AdamW optimizer and gradient norm clipping during training. The models are trained for a few epochs with a batch size of 256 segments. [More Information Needed]
5. The training hyperparameters are not provided in the references. [More Information Needed]

Please note that the above information is based on the provided references, and additional information may be required to provide a complete and accurate model card description.

- **Developed by:** Alec Radford; Jong Wook Kim; Tao Xu; Greg Brockman; Christine Mcleavey; Ilya Sutskever
- **Funded by:** The model openai/whisper-small.en is a general-purpose speech recognition model trained on a large dataset of diverse audio. It is a multitasking model that can perform multilingual speech recognition, speech translation, and language identification. The model is a part of the Whisper project developed by OpenAI.

The funding details for the Whisper project and the model openai/whisper-small.en are not mentioned in the provided references. [More Information Needed]
- **Shared by:** The contributors who made the model openai/whisper-small.en available online as a GitHub repo are OpenAI.
- **Model type:** The model openai/whisper-small.en is a general-purpose speech recognition model trained on a large dataset of diverse audio, utilizing a multitasking approach for multilingual speech recognition, speech translation, and language identification.
- **Language(s):** The model openai/whisper-small.en uses natural human language for speech recognition and transcription without significant standardization, relying on sequence-to-sequence models to predict the raw text of transcripts.
- **License:** The license being used for the model openai/whisper-small.en is the MIT License. You can find further details about the license [here](https://github.com/openai/whisper/blob/main/LICENSE).
- **Finetuned from model:** The model openai/whisper-small.en is fine-tuned from another model, but the name and link to that base model are not provided in the given references. [More Information Needed]
### Model Sources

- **Repository:** https://github.com/openai/whisper
- **Paper:** https://arxiv.org/pdf/2212.04356.pdf
- **Demo:** To find the link to the demo of the model openai/whisper-small.en, we can refer to the references. However, there is no direct reference to the demo of the model openai/whisper-small.en in the given references. 

Therefore, the link to the demo of the model openai/whisper-small.en is [More Information Needed].
## Uses

### Direct Use

The model openai/whisper-small.en can be used without fine-tuning, post-processing, or plugging into a pipeline by utilizing the `whisper.transcribe()` method. This method takes an audio file as input and returns the transcribed text. Here is an example code snippet:

```python
import whisper

model = whisper.load_model("openai/whisper-small.en")
result = model.transcribe("audio.mp3")
print(result["text"])
```

The `transcribe()` method internally reads the entire audio file and processes it with a sliding 30-second window. It performs autoregressive sequence-to-sequence predictions on each window to generate the transcriptions.

### Downstream Use

The model openai/whisper-small.en can be fine-tuned for various speech processing tasks or plugged into a larger ecosystem or app. Fine-tuning the model involves training it on a specific dataset to improve its performance on a task.

To fine-tune the model for a specific task, a simple format is used to specify the task and conditioning information as a sequence of input tokens to the decoder. For example, if the task is transcription, the input audio signal can be passed to the model, and the model will generate the corresponding text transcript.

Here is a code snippet that demonstrates how the model can be fine-tuned for transcription:

```python
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

# Load the pre-trained model
model_name = "openai/whisper-small.en"
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example audio input
audio_input = "path/to/audio.wav"

# Convert audio to text transcript
input_ids = tokenizer(audio_input, return_tensors="tf").input_ids
outputs = model.generate(input_ids)
transcript = tokenizer.decode(outputs[0])

print(transcript)
```

This code snippet demonstrates how to use the fine-tuned model to convert an audio input into a text transcript. However, specific details about the fine-tuning process, such as the dataset used, hyperparameters, and training setup, are not provided in the references, so further information would be needed to fine-tune the model effectively.

In terms of plugging the model into a larger ecosystem or app, it can be used as a speech processing component. For example, it can be integrated into a transcription service or an automatic speech recognition system. The model can take an audio input and generate the corresponding text output, enabling it to be used in various speech-related applications.

[More Information Needed]

### Out-of-Scope Use

Model Card: openai/whisper-small.en

## Model Description

The model openai/whisper-small.en is a speech recognition model developed by OpenAI. It is built using the Whisper architecture and trained on a large and diverse supervised dataset, focusing on zero-shot transfer for improved robustness.

## Potential Misuse and Guidelines

It is important to consider the potential misuses of the openai/whisper-small.en model and provide guidelines to prevent harmful or unethical use. While the following guidelines are not exhaustive, they highlight some key considerations:

1. **Avoid generating misleading or malicious transcriptions**: Users ought not to use the model to intentionally generate transcriptions that are misleading, false, or malicious. The model's output should not be manipulated to spread misinformation or harm individuals or groups.

2. **Do not use the model for unauthorized surveillance**: Users should not employ the model for unauthorized surveillance purposes, such as transcribing private conversations without consent. Respecting privacy rights is crucial, and the model should not be used to violate them.

3. **Exercise caution with sensitive information**: The model should not be used to transcribe or process sensitive or confidential information without appropriate security measures. Users should be cautious when dealing with protected data, such as personally identifiable information or classified content.

4. **Avoid excessive reliance without human review**: It is important to remember that the model's output may contain errors or inaccuracies. Users should not solely rely on the model's transcriptions without human review and verification, especially in critical or high-stakes applications.

5. **Consider potential biases and fairness**: The model may reflect biases present in the training data, including language biases and cultural biases. Users should be mindful of these biases and take appropriate steps to mitigate any potential discriminatory or unfair outcomes.

6. **Comply with legal and ethical standards**: Users should adhere to applicable laws, regulations, and ethical guidelines when using the model. This includes respecting intellectual property rights, privacy laws, and maintaining ethical standards in the use of AI technologies.

It is important to note that these guidelines are intended to provide a general framework for responsible use, but each specific use case may require additional considerations and precautions.

For further information about the openai/whisper-small.en model, please refer to the [model card](https://github.com/openai/whisper/blob/main/model-card.md) and the associated [paper](https://arxiv.org/abs/2212.04356).

### Bias, Risks, and Limitations

The known or foreseeable issues stemming from the model openai/whisper-small.en include:

1. Perception-related errors: While larger models have shown progress in reducing perception-related errors, there are still remaining errors, particularly in long-form transcription. These errors are a combination of failure modes of seq2seq models, language models, and text-audio alignment. They can include problems such as repeat loops, non-human/perceptual errors, and hallucination of unrelated transcripts.

2. Poor performance on lower-resource languages: The speech recognition performance of Whisper is still quite poor on many languages. The amount of training data for a language strongly predicts performance, but the pre-training dataset for Whisper is currently English-heavy due to biases in data collection. This limits performance on other languages.

3. Unclear benefits of encoder and decoder: It is currently unclear to what degree the benefits of Whisper stem from training its encoder, decoder, or both. Further research is needed to study the impact of different design components of Whisper and how they affect performance.

4. Lack of generalizability and robustness: Machine learning models, including Whisper, have been shown to lack generalization and robustness. They can make mistakes when evaluated in slightly different settings or datasets. Whisper's robustness properties need further investigation to understand its performance in different scenarios.

5. Speaker name prediction: Whisper has a tendency to transcribe plausible but incorrect guesses for the names of speakers. This is because many transcripts in the pre-training dataset include speaker names, encouraging the model to predict them. However, this information is rarely inferable from the audio alone.

6. Brittle and lack of robustness: While Whisper matches or outperforms humans on certain datasets, supervised models still make more errors than humans on other datasets, demonstrating brittleness and lack of robustness. Closing the gap to human-level robustness is a challenge.

These issues highlight the need for further research and improvement to address the limitations and potential harms of the model openai/whisper-small.en.

### Recommendations

Based on the references provided, the following recommendations can be made with respect to the foreseeable issues about the model openai/whisper-small.en:

1. **Improved decoding strategies**: Larger models have shown progress in reducing perception-related errors such as confusing similar-sounding words. However, stubborn errors remain, particularly in long-form transcription. These errors include problems like repeat loops and hallucinations. Further research can be done on fine-tuning the model using high-quality supervised datasets and reinforcement learning to optimize decoding performance.

2. **Increase training data for lower-resource languages**: Whisper's speech recognition performance is poor in many languages. It has been observed that the amount of training data for a language strongly influences performance. Since the pre-training dataset is English-heavy, it is recommended to increase the training data for other languages to improve performance.

3. **Addressing hallucination and incorrect guesses**: The model has a tendency to generate incorrect guesses for speaker names. This is because many transcripts in the pre-training dataset include speaker names, which encourages the model to predict them. Techniques or heuristics should be developed to address this issue and prevent the model from generating incorrect information.

4. **Robustness to distribution shift and perturbations**: It is important to investigate how effectively the model transfers and how robust it is to distribution shift and perturbations. Research on the model's robustness should be conducted to ensure its performance in different scenarios.

5. **Removal of machine-generated transcripts**: Many transcripts on the internet are not human-generated but the output of existing ASR systems. Training on datasets with mixed human and machine-generated data can significantly impair the performance of the model. Therefore, heuristics should be developed to detect and remove machine-generated transcripts from the training dataset.

6. **Code release for easy comparison and performance evaluation**: The code for the model's text normalizer should be released to facilitate easy comparison with other speech recognition systems and to enable the study of its performance in out-of-distribution settings.

Overall, further research and improvements are needed to address the remaining errors, improve performance in lower-resource languages, handle hallucination and incorrect guesses, ensure robustness to distribution shift, remove machine-generated transcripts, and provide code for easy comparison and performance evaluation.

## Training Details

### Training Data

The training data for the model openai/whisper-small.en consists of raw transcripts without significant standardization, and the model learns to map between utterances and their transcribed form using sequence-to-sequence models. The data undergoes a filtering pass to remove low-quality transcripts, including partially transcribed or misaligned transcripts, as well as machine-generated captions. For more information, please refer to References 1, 4, and 5.

### Training Procedure

#### Preprocessing

The preprocessing steps for the data of the model openai/whisper-small.en are as follows:

1. Tokenization: The data is tokenized using a sequence-to-sequence Transformer model. The tokens used include special tokens like <|endoftranscript|>, <|startoftranscript|>, <|nospeech|>, and task specifiers or classification targets.

2. Resizing/Rewriting: The data goes through a resizing or rewriting process where aspects that are difficult to predict from audio signals, such as complex punctuation, formatting whitespace, or capitalization, are removed or normalized. This is done to simplify the speech recognition pipeline.

3. Data Pre-processing: Unlike traditional speech recognition approaches, the Whisper models are trained to predict the raw text of transcripts without significant standardization. The models rely on the expressiveness of sequence-to-sequence models to learn the mapping between utterances and their transcribed form.

4. Text Normalization: For English texts in different styles, a best-effort attempt is made to normalize them into a standardized form. This process is designed to penalize word errors caused by mistranscribing a word, rather than formatting or punctuation differences.

Unfortunately, there is no reference to specific code blocks for these preprocessing steps. [More Information Needed].

#### Training Hyperparameters

The training hyperparameters for the model openai/whisper-small.en are as follows:

- Training method: Data parallelism across accelerators
- Precision: FP16 with dynamic loss scaling
- Activation checkpointing: Yes
- Optimizer: AdamW
- Gradient norm clipping: Yes
- Learning rate decay: Linear decay to zero after a warmup over the first 2048 updates
- Batch size: 256 segments
- Training duration: 2 20 updates
- Data augmentation: None
- Regularization: None

For more detailed training hyperparameters, please refer to Appendix F in the reference document.

Please note that the model has a tendency to transcribe plausible but almost always incorrect guesses for the names of speakers due to the presence of speaker names in the pre-training dataset. The model is trained to predict these names but lacks the necessary context to infer them accurately.

To transcribe speech using the model, you can use the following Python code:

```python
import whisper

model = whisper.load_model("base")
result = model.transcribe("audio.mp3")
print(result["text"])
```

This code uses the `transcribe()` method of the model to process the audio with a sliding 30-second window and perform autoregressive sequence-to-sequence predictions on each window.

Unfortunately, there is no specific mention of the training hyperparameters for the model openai/whisper-small.en in the provided references.

#### Speeds, Sizes, Times

The openai/whisper-small.en model is a part of the Whisper system, which is a single robust speech processing system designed to work reliably without the need for dataset-specific fine-tuning. The model has been trained on a diverse set of existing speech processing datasets to ensure good generalization across domains, tasks, and languages.

Unfortunately, the provided references do not provide detailed information about the throughput, start or end time, or checkpoint sizes specifically for the openai/whisper-small.en model. Therefore, I would need more information to provide a specific answer to this question.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model openai/whisper-small.en evaluates its performance on the following benchmarks and datasets:

1. LibriSpeech test-clean: The zero-shot Whisper model achieves an average relative error reduction of 55.2% when evaluated on this dataset compared to a supervised LibriSpeech model. (Reference: 3)

2. Other speech recognition datasets: The zero-shot Whisper model achieves competitive performance when evaluated on datasets other than LibriSpeech test-clean. (Reference: 5)

3. Common Voice 15 and Fleurs datasets: The performance breakdown of the `large-v3` and `large-v2` models by language is evaluated using WERs or CERs on these datasets. (Reference: 6)

4. Multilingual LibriSpeech (MLS) corpus: The test splits from each language in the MLS corpus are used for evaluation. (Reference: 7)

5. Fleurs dataset: The zero-shot performance of Whisper is evaluated for language identification on this dataset, where it underperforms the supervised state-of-the-art by 13.6%. (Reference: 11)

Please note that additional benchmarks and datasets may exist, but the provided references do not mention them.

#### Factors

The foreseeable characteristics that will influence how the model openai/whisper-small.en behaves include:

1. **Domain and Context**: The model's performance may vary across different domains and contexts. The model's training data includes a wide set of existing speech processing datasets, which helps it generalize well across domains, tasks, and languages. However, the specific characteristics of a given domain or context may still impact the model's behavior.

2. **Population Subgroups**: The model's performance may differ across population subgroups. The model has been evaluated on various datasets, but it's important to disaggregate the evaluation results across factors such as language, cultural background, and demographic characteristics to uncover any potential disparities in performance.

In order to fully understand the model's behavior and potential biases, a comprehensive evaluation that considers domain and context as well as population subgroups is necessary. This evaluation should involve careful analysis and testing to ensure that the model performs well across diverse scenarios and does not disproportionately favor or disadvantage any particular groups.

#### Metrics

The metrics used for evaluation of the model openai/whisper-small.en in light of tradeoffs between different errors are the Word Error Rate (WER) and Character Error Rate (CER). These metrics are based on string edit distance and penalize all differences between the model's output and the reference transcript, including innocuous differences in transcript style. However, the evaluation also takes into account non-semantic differences and the impact of text normalization to minimize penalization of these differences. The evaluation methodology includes extensive standardization of text before calculating WER to address these tradeoffs. For further details, please refer to [the paper](https://arxiv.org/abs/2212.04356).

### Results

The evaluation results of the model openai/whisper-small.en based on the Factors and Metrics are not provided in the given references. [More Information Needed]

#### Summary

The evaluation results for the model openai/whisper-small.en are summarized as follows:

The model achieves a Word Error Rate (WER) of 6.7 on the LibriSpeech test-clean dataset. It is important to note that this model has only 39 million parameters, which makes it the smallest zero-shot Whisper model. Despite its small size, it performs competitively with the best supervised LibriSpeech model when evaluated on other datasets. In fact, when compared to a human, the best zero-shot Whisper models roughly match their accuracy and robustness.

The model outperforms other open-source models and commercial ASR services on most datasets, especially on datasets with uncommon words. The performance of the model is summarized in Figure 6 of the referenced paper.

The goal of the Whisper model is to develop a single robust speech processing system that can generalize well across domains, tasks, and languages without the need for dataset-specific fine-tuning. It achieves this goal by reusing a wide set of existing speech processing datasets for evaluation.

For a detailed breakdown of the improvement in robustness and performance, please refer to Table 2 in the referenced paper.

Overall, the evaluation results demonstrate that the openai/whisper-small.en model performs well on various datasets and shows promise in achieving high-quality results without the need for extensive fine-tuning.

For more detailed information, please refer to the references provided.

## Model Examination

The model openai/whisper-small.en is an experimental speech recognition model developed by OpenAI. It is trained using a large pre-training dataset that is primarily English-centric. The model's performance on speech recognition is still poor for many languages, indicating the need for increased training data for lower-resource languages.

The model has been designed to transcribe audio segments by either transcribing the first or last few words or generating a completely unrelated transcript. However, there are still errors in the decoding process, including repeat loops and non-human/perceptual errors.

The model's robustness properties have been a focus of study, particularly in terms of zero-shot transfer performance. While fine-tuning has not been extensively studied, it is expected that fine-tuning on high-quality supervised speech data can further improve the model's results.

The model also exhibits a tendency to transcribe plausible but incorrect guesses for the names of speakers, which can be attributed to the pre-training dataset including speaker names in transcripts.

For more information on the model, including example usages and third-party extensions, the OpenAI Whisper team suggests referring to the "Show and tell" category in the Discussions section.

In terms of explainability/interpretability, the provided references do not contain specific information about work done on this aspect for the model openai/whisper-small.en. Therefore, [More Information Needed].

## Environmental Impact

- **Hardware Type:** The hardware type that the model openai/whisper-small.en is trained on is not explicitly mentioned in the provided references. [More Information Needed]
- **Software Type:** The model openai/whisper-small.en is trained on a multitasking model that can perform multilingual speech recognition, speech translation, and language identification.
- **Hours used:** The amount of time used to train the model openai/whisper-small.en is not mentioned in the provided references. [More Information Needed]
- **Cloud Provider:** The cloud provider that the model openai/whisper-small.en is trained on is not mentioned in the provided references. [More Information Needed]
- **Carbon Emitted:** The amount of carbon emitted when training the model openai/whisper-small.en is not mentioned in the provided references. [More Information Needed]
## Technical Specification

### Model Architecture and Objective

The model openai/whisper-small.en is based on an encoder-decoder Transformer architecture. The architecture consists of pre-activation residual blocks for the encoder and learned position embeddings for the decoder. The encoder and decoder have the same width and number of transformer blocks.

The objective of the model is to perform speech recognition. It is trained on a large-scale supervised pre-training dataset, which includes audio samples re-sampled to 16,000 Hz and transformed into an 80-channel logmagnitude Mel spectrogram representation. The model predicts the raw text of transcripts without significant standardization, relying on the expressiveness of sequence-to-sequence models to learn the mapping between utterances and their transcriptions.

[More Information Needed]

### Compute Infrastructure

The compute infrastructure for the model openai/whisper-small.en is not directly specified in the provided references. [More Information Needed].

## Citation

```
@misc{alec-robust,
    author = {Alec Radford and
              Jong Wook Kim and
              Tao Xu and
              Greg Brockman and
              Christine Mcleavey and
              Ilya Sutskever},
    title  = {Robust Speech Recognition via Large-Scale Weak Supervision},
    url    = {https://arxiv.org/pdf/2212.04356.pdf}
}
```

