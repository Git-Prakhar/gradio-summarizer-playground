'''
This contains the entire code in one file.
This file won't get updated frequently, as it is not the main file.
It is provided for reference and to run the entire application without needing to split it into multiple files.
'''

import gradio as gr
import torch
from transformers import pipeline
import json

# -------- Functions --------

# -- Model Loading Functions --
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

# -- Model Management Functions --
# -------- Add a new model to the list --------
def add_model(model_name, model_link):
    try:
        # -------- Load existing models from the JSON file --------
        with open(model_file, 'r') as file:
            models = json.load(file)
    except FileNotFoundError:
        # -------- If the file does not exist, initialize an empty dictionary --------
        models = []

    # -------- Validate the model name and link --------
    if model_name.strip() == "" or model_link.strip() == "":
        return "Enter all the fields"
    
    # -------- Check if the custom model name already exists in the list --------
    if model_name not in models:
        # -------- Check if the huggingface model name already exists in the list --------
        for model_value in models.values():
            if model_value == model_link.strip():
                return "Model link already exists in the list"
            
        # -------- Add the new model to the list --------
        models[model_name] = model_link

        try:
            # -------- Save the updated models list back to the JSON file --------
            with open(model_file, 'w') as file:
                json.dump(models, file, indent=4)

            # -------- Return success message --------
            return f"Model {model_name} added successfully"
        except Exception as e:
            # -------- Handle errors while saving the model --------
            return f"Error while saving the model: {e}"
    else:
        return "Model already exists in the list" # Model name already exists in the list
    

# -------- Load the list of models from the JSON file --------
def load_models():
    try:
        # -------- Attempt to read the models from the JSON file --------
        with open(model_file, 'r') as file:
            models = json.load(file)
            return models
    except FileNotFoundError:
        return {} # Return an empty dictionary if the file does not exist
    except json.JSONDecodeError:
        return {} # Return an empty dictionary if the file is not a valid JSON
    

# -------- Delete a specified model from the list --------
def delete_model(model_name):
    try:
        # -------- Load existing models from the JSON file --------
        with open(model_file, 'r') as file:
            models = json.load(file)
    except FileNotFoundError:
        return "Model file not found" # If the file does not exist, return an error message
    
    # -------- Check if the model exists in the list and delete it --------
    try:
        if model_name in models:
            del models[model_name]
            with open(model_file, 'w') as file:
                json.dump(models, file, indent=4) # Save the updated models list back to the JSON file
            return f"Model {model_name} deleted successfully"
        else:
            return "Model not found in the list"
    except Exception:
        return f"Error while deleting the model and saving the list" # Handle any errors that occur during deletion and saving

# -- Gradio Functions --
# -------- Load the list of models from the JSON file --------
def load_model_list():
    models = load_models()
    model_names = list(models.keys()) if models else []
    model_names.append("Custom Model")
    return model_names

# -------- For dropdown update on refresh button click --------
def refresh_model_dropdown():
    return gr.update(choices=load_model_list())

# -------- Toggle custom model textbox on dropdown selection change --------
def toggle_custom_model(model_choice):
    return gr.update(visible=(model_choice == "Custom Model"))


# -------- Toggle between Add and Delete Modals --------
def toggle_modal(primary_modal_state, secondary_modal_state):
    new_primary = not primary_modal_state
    new_secondary = False

    print(f"[INFO] Current modal states: primary={primary_modal_state}, secondary={secondary_modal_state}")
    print(f"[INFO] Toggling modals: primary={new_primary}, secondary={new_secondary}")
    
    return gr.update(visible=new_primary), gr.update(visible=new_secondary), new_primary, new_secondary


# -------- Variables --------
devices = load_devices()
print("[INFO] Available Devices:", devices)
model_file = "models.json"
selectedModel = {"name": "NA"}
modelCache = {}

# -------- Main Gradio App --------

with gr.Blocks() as demo:
    #  -------- App Title and Instructions --------
    gr.Markdown("# Welcome to the Text Summarization App!")
    gr.Markdown("""
1. In case of any issues or information, read the [README](README.md) file.
2. In case of issues not covered in the README, please open an issue on the [GitHub Repository](github.com)
3. I'll try my best to resolve it as soon as possible.
""")

    # --------- Model Management Section ---------
    with gr.Row():
        add_option_btn = gr.Button("Add a model to the list")
        delete_option_btn = gr.Button("Delete from model list")
    add_modal_area = gr.Column(visible=False)
    delete_modal_area = gr.Column(visible=False)

    # --------- Delete Model Modal ---------
    with delete_modal_area:
        gr.Markdown("### Delete a Model")
        with gr.Row():
            with gr.Column(scale=1):
                delete_model_name = gr.Dropdown(
                    choices=load_model_list(),
                    label="Select Model to Delete"
                )
                refresh_btn = gr.Button("Refresh Model List")
            model_delete_status = gr.Textbox(label="Model Deletion Status", interactive=False)
        refresh_btn.click(
            fn=refresh_model_dropdown,
            outputs=delete_model_name
        )

        delete_model_btn = gr.Button("Delete Model")

        delete_model_btn.click(
            fn=delete_model,
            inputs=delete_model_name,
            outputs=model_delete_status
        )

    # --------- Add Model Modal ---------
    with add_modal_area:
        gr.Markdown("### Add a Custom Model")
        custom_model_name = gr.Textbox(label="Enter Custom Model Name", placeholder="E.g., My Pegasus Model")
        custom_model_link = gr.Textbox(label="Paste Hugging Face Model Name", placeholder="E.g., google/pegasus-cnn_dailymail")
        model_add_status = gr.Textbox(label="Model Addition Status", interactive=False)
        add_model_btn = gr.Button("Add Model")

        add_model_btn.click(
            fn=add_model,
            inputs=[custom_model_name, custom_model_link],
            outputs=model_add_status
        )
    
    # -------- Initialize modal states --------
    add_modal_state = gr.State(value=False)
    delete_modal_state = gr.State(value=False)

    # -------- Add and Delete Option Buttons Functionality --------
    add_option_btn.click(toggle_modal, outputs=[add_modal_area, delete_modal_area, add_modal_state, delete_modal_state], inputs=[add_modal_state, delete_modal_state])
    delete_option_btn.click(toggle_modal, outputs=[delete_modal_area, add_modal_area, delete_modal_state, add_modal_state], inputs=[delete_modal_state, add_modal_state])
    

    # ------------------ Model Selection ------------------
    gr.Markdown("## Select a Model to Load")
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=load_model_list(),
            label="Model"
        )

        custom_model_inp = gr.Textbox(
            label="Custom Model Name",
            placeholder="Enter custom model name here...",
            interactive=True,
            visible=False
        )

        device_dropdown = gr.Dropdown(
            devices,
            label="Load on:"
        )
    with gr.Row():
        refresh_model_btn = gr.Button("Refresh Model List")
        load_model_btn = gr.Button("Load Model")

    selected_model_display = gr.Textbox(label="Selected Model", interactive=False) # Display the output after loading the model

    # -------- Toggle custom model input visibility based on dropdown selection --------
    model_dropdown.change(
        fn=toggle_custom_model,
        inputs=model_dropdown,
        outputs=custom_model_inp
    )

    # -------- Refresh the model list when clicking the refresh button --------
    refresh_model_btn.click(
        fn=refresh_model_dropdown,
        outputs=model_dropdown
    )

    # -------- Load the selected model or custom model when clicking the load button --------
    load_model_btn.click(
        fn=load_model,
        inputs=[model_dropdown, custom_model_inp, device_dropdown],
        outputs=selected_model_display
    )

    # ------------------ Summarization ------------------
    gr.Markdown("## Summarize Your Text")

    with gr.Row():
        prompt = gr.Textbox(label="Enter Text", placeholder="Type or paste text to summarize...", lines=5)

    with gr.Row():
        min_len = gr.Number(label="Min Length", value=50, step=1, minimum=10)
        max_len = gr.Number(label="Max Length", value=150, step=1, minimum=100)

    summarize_btn = gr.Button("Summarize!")
    summary_output = gr.Textbox(label="Summarized Text")

    # -------- Summarize the prompt when clicking the summarize button --------
    summarize_btn.click(
        fn=summarize_prompt,
        inputs=[prompt, max_len, min_len],
        outputs=summary_output
    )

demo.launch()