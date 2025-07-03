'''
This module provides functionality to manage a list of models for a summarization application.
It allows adding, loading, and deleting models from a JSON file.

json file : `models.json`

functions include:
- `add_model`: Adds a new model to the list if it does not already exist.
- `load_models`: Loads the list of models from the JSON file.
- `delete_model`: Deletes a specified model from the list.
'''

import json
model_file = "models.json"

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