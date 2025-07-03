'''
This is the main file for the Text Summarization App using Gradio.
It provides a user interface for managing models, loading them, and summarizing text.
'''

# -------- Library Imports --------
import gradio as gr
from modelFn import load_model, summarize_prompt, load_devices
from modelOptions import add_model, load_models, delete_model

# -------- Load available devices --------
devices = load_devices()
print("[INFO] Available Devices:", devices)

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

# -------- Main Gradio App --------
with gr.Blocks() as demo:
    #  -------- App Title and Instructions --------
    gr.Markdown("# Welcome to Summarizer Playground!")
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