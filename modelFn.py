'''
This module provides functions to load and manage summarization models using the Hugging Face Transformers library.
It includes functionalities to load predefined models, handle custom models, and perform text summarization.

functions include:
- `load_model`: Loads a specified model or a custom model from Hugging Face.
- `summarize_prompt`: Summarizes a given text prompt using the loaded model.
- `load_devices`: Lists available devices (CPU and GPUs) for model inference.
'''

import torch
from transformers import pipeline
from modelOptions import load_models

# -------- Selected Model and Cache --------
selectedModel = {"name": "NA"}
modelCache = {}

def load_model(model_name, custom_model_name, device):
    models = load_models() # Load predefined models from the JSON file

    # -------- Check if the custom model name is valid --------
    if model_name == "Custom Model" and custom_model_name.strip() == "":
        return "Please enter a valid custom model name."

    # -------- Check if the model is already selected and loaded --------
    if selectedModel['name'] == model_name and model_name in modelCache:
        return f"Model {model_name} is already loaded."
    
    try:
        # -------- Selecting user selected device --------
        devices = load_devices()
        device_index = -1
        if device != "CPU":
            device_index = devices.index(device) - 1
    except Exception as e:
        return f"Error while selecting hardware device: {e}"
    
    try:
        # -------- Setting up model's Hugging Face name --------
        model_huggingface_name = models.get(model_name, None) if model_name != "Custom Model" else custom_model_name.strip()
        if model_huggingface_name is None:
            return f"Model {model_name} not found in the predefined models. Please add it first."
    except Exception as e:
        return f"Error while setting up model name: {e}"
    
    try:
        # -------- Loading the model using Hugging Face pipeline if not already cached --------
        if model_name in modelCache:
            pipe = modelCache[model_name]
        else:
            pipe = pipeline("summarization", model=model_huggingface_name, device=device_index)
            modelCache[model_name] = pipe # Cache the model for future use

    except Exception as e:
        # -------- Handle errors during model loading --------
        if model_name == "Custom Model":
            return f"Error while searching and loading custom model {custom_model_name}: {e}"
        return f"Error while searching and loading predefined model {model_name}: {e}"

    # -------- Updating the selected model name --------
    selectedModel["name"] = model_name

    # -------- Return success message --------
    return f"Loaded Model: {model_name} on {device}" if model_name != "Custom Model" else f"Loaded Custom Model: {custom_model_name} on {device}"
    

def summarize_prompt(prompt, max_len, min_len):
    
    # -------- Validate the input prompt --------
    if not prompt or len(prompt.strip()) == 0:
        return "Please enter a valid text to summarize."
    if max_len < min_len:
        return "Max length must be greater than or equal to Min length."
    
    # -------- Check if a model is selected and loaded --------
    if selectedModel['name'] == "NA":
        return f"Please load a model"
    if selectedModel['name'] not in modelCache:
        return f"Model {selectedModel['name']} not found in cache. Please load the model first."
    
    # -------- Get the model from cache and summarize the prompt --------
    pipe = modelCache[selectedModel['name']]
    try:
        summary = pipe(prompt, max_length=max_len, min_length=min_len)

        # -------- Check if the summary is empty --------
        if not summary or len(summary) == 0:
            return "No summary generated. Please check the input text or model configuration."
        
        # -------- Return the summary text --------
        return summary[0]['summary_text']
    
    except Exception as e:
        # -------- Handle errors during summarization --------
        return f"Error during summarization: {e}"

def load_devices():
    devices = ["CPU"]

    # -------- Check if CUDA is available and list all available GPUs --------
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            devices.append(name)
    
    return devices