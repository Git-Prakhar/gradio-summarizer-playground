# Summarizer Playground
Welcome to the Text Summarization Gradio App! Choose your model, input your text, and get a summary. <br /><br />

## Description :
This app is made solely using the [Gradio](https://gradio.app/) library, which provides a simple interface to interact with models and thus the app is not so user-friendly.<br /><br />
This app only allows you to load models from [hugging face's summarization tasked models](https://huggingface.co/models?pipeline_tag=summarization), making it totally free to use.<br /><br />
The models are loaded into your local environment using your CPU or cuda-enabled GPU (if available), to summarize the text you provide, locally.<br /><br />
<b>Note: Load models that are compatible with your device's CPU/GPU and RAM capacity.</b><br /><br />

## Table of Contents :
- [Installation](#installation)
- [Limitations and Issues](#limitations-and-issues)
- [Issues and their Solutions](#issues-and-their-solutions)

### Installation :
#### 1. Prequisites:
- Python 3.7 or higher
- Libraries:

```bash
pip install gradio transformers torch
```
#### 2. Clone the repository
```bash
git clone giturl
```
#### 3. Navigate to the cloned directory
```bash
cd Summarizers
```
#### 4. Run the `index.py` file
```bash
python index.py
```


### Limitations and Issues :
1. The app is hosted on your local machine, so it is fully dependent on your local resources.
2. The app is not so user-friendly, as it is made solely using the Gradio library.
3. If a model is not compatible with your device's CPU/GPU and RAM capacity, it may not load properly or may cause the app to crash.
4. The app does not provide a live loader for the models, so you have to wait for the model to load before you can use it (it can be quite unpredictable based on the model size and your device's resources).

### Issues and their Solutions :
#### 1. **Model not loading properly**: 
- Ensure that the model you are trying to load is compatible with your device's CPU/GPU and RAM capacity.
- If the model is too large, try loading a smaller model or using a device with more resources.
- Ensure that you have entered the correct hugging face model name (Eg: `facebook/bart-large-cnn`).
#### 2. **GPU not available**: 
- Ensure that you have a CUDA-enabled GPU and the necessary drivers installed.
- Check if the `torch.cuda.is_available()` returns `True` in the console.
- If it returns `False`, and you have a CUDA-enabled GPU, ensure that you have installed the correct version of PyTorch with CUDA support.
    - Try running `torch.version()` and check if it returns the correct version of PyTorch with CUDA support Eg: `1.12.0+cu113`.
    - If it doesn't then uninstall PyTorch and reinstall it with the correct version.
#### 3. **No predefined models available**: 
- Ensure that you have a file in your base directory named `models.json` with the following format
```json
{
"Facebook -> bart-large-xsum": "facebook/bart-large-xsum",
"Facebook -> bart-large-cnn": "facebook/bart-large-cnn",
"Google -> pegasus-cnn_dailymail": "google/pegasus-cnn_dailymail",
"Falconsai -> text_summarization": "Falconsai/text_summarization",
"sshleifer -> distilbart-cnn-12-6": "sshleifer/distilbart-cnn-12-6"
}
```
#### 4. **Taking too long to load the model**: 
- Ensure that you have a stable internet connection.
- If the model is too large, it may take some time to load. Try loading a smaller model or using a device with more resources.