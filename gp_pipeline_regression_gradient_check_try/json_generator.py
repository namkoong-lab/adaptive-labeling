import json
import re
import os
from copy import deepcopy
import pickle as pkl
import pandas as pd






def fill_json_with_file_combinations(data, filecombinations, model_n_event, model_n_individual, model_max_length):
    data = deepcopy(data)    
    data['parameters']['files']['values'] = filecombinations
    data['parameters']['n_events']['value'] = model_n_event
    data['parameters']['n_individual']['value'] = model_n_individual
    data['parameters']['max_seq_length']['value'] = int(model_max_length)
    return data

def process_single_training_group(training_group, json_template, length_pattern = r"\_ln\_(\d+)\_"):

    filecombinations = training_group[['training', 'validation', 'testing']].to_dict()
    model_n_event = int(training_group['model_n_event'])
    model_n_individual = int(training_group['model_n_individual'])
    training_group_training_length = int(re.findall(length_pattern, training_group['training'])[0])
    training_group_validation_length = int(re.findall(length_pattern, training_group['validation'])[0])
    training_group_testing_length = int(re.findall(length_pattern, training_group['testing'])[0])
    assert training_group_training_length == training_group_validation_length == training_group_testing_length, "All the files should have the same sequence length"
    complete_json = fill_json_with_file_combinations(json_template, [filecombinations], model_n_event, model_n_individual, training_group_training_length)
    return complete_json

def process_multiple_training_groups(training_group, json_template, length_pattern = r"\_ln\_(\d+)\_"):
    filecombinations = list(training_group[['training', 'validation', 'testing']].T.to_dict().values())
    model_n_event = int(training_group['model_n_event'].values[0])
    model_n_individual = int(training_group['model_n_individual'].values[0])
    # print(training_group)
    
    training_group_training_length = pd.unique(training_group['training'].apply(lambda x: int(re.findall(length_pattern, x)[0])))
    training_group_validation_length = pd.unique(training_group['validation'].apply(lambda x: int(re.findall(length_pattern, x)[0])))
    training_group_testing_length = pd.unique(training_group['testing'].apply(lambda x: int(re.findall(length_pattern, x)[0])))
    # print(training_group_training_length, training_group_validation_length, training_group_testing_length)
    assert training_group_training_length.shape[0] == training_group_validation_length.shape[0] == training_group_testing_length.shape[0] == 1, "All the files should have only one sequence length"
    assert training_group_training_length[0] == training_group_validation_length[0] == training_group_testing_length[0], "All the files should have the same sequence length"
    complete_json = fill_json_with_file_combinations(json_template, filecombinations, model_n_event, model_n_individual, training_group_training_length[0])
    return complete_json
    
    
    




# This is a simple copy of config_sweep_warmup.json

def generate_warmup_json(
        json_prefix = "config_sweep_warmup",
        program_name = "main.py",
        project_name = "Untitled_Project",
        simulation_log_path = "simulation_logs.pkl",
        json_folder_path = "json_files",
        cluster_path = ""
):
    if not os.path.exists(json_folder_path):
        os.makedirs(json_folder_path)    
    generated_json_files = []
    json_template = {
        "method": "grid",
        "metric": {
            "name": "validation_loss",
            "goal": "minimize"
        },
        "parameters": {
            "n_events": {
            },
            "n_individual":{
            },
            "d_embedding_individual":{
                "values": [64]
            },
            "d_embedding_event": {
                "values": [64]
            },
            "d_embedding_time": {
                "values": [64]
            },
            "max_seq_length": {
            },
            "nhead": {
                "values": [4]
            },
            "d_hid": {
                "values": [512]
            },
            "nlayers": {
                "values": [8]
            },
            "dropout_trans": {
                "values": [0.0]
            },
            "dropout_pos": {
                "value": 0.1
            },
            "w_individual": {
                "value": 1.0
            },
            "w_event": {
                "value": 1.0
            },
            "w_time": {
                "value": 1.0
            },
            "grad_clip": {
                "values": [1.0]
            },
            "weight_decay": {
                "values": [1e-6]
            },
            "learning_rate": {
                "values": [5e-4]
            },
            "scheduler_step_size": {
                "value": 1
            },
            "scheduler_gamma": {
                "values": [2.0]
            },
            "epochs": {
                "values": [5000]
            },
            "batch_size": {
                "values": [512]
            },
            "files":{
            },
            "save_dir": {
                "value": os.path.join(cluster_path, "checkpoints/")
            },
            "shuffle": {
                "values": [False]
            },
            "load":{
                "value": True
            },
            "load_project_name":{
                "value": ""
            },
            "load_artifact_name":{
                "value": ""
            },
            "save":{
                "value": True
            },
            'save_project_name':{
                "value": ""
            },
            'save_artifact_name':{
                "value": ""
            }
        }
    }

    json_loaded = []
    length_pattern = r"\_ln\_(\d+)\_"
    with open(simulation_log_path, 'rb') as f:
        simulation_log = pkl.load(f)
    simulation_log_df = pd.DataFrame(simulation_log)
    # print('simulattion_log_df.columns:', simulation_log_df.columns)
    if simulation_log_df.ndim > 1:
        groups = simulation_log_df.groupby(['model_n_event', 'model_n_individual', 'max_sequence_length']).groups
        for group_name, group_indexes in groups.items():
            training_group = simulation_log_df.loc[group_indexes].copy()
            if training_group.ndim > 1:
                json_loaded.append(process_multiple_training_groups(training_group, json_template, length_pattern))
                # print(process_multiple_training_groups(training_group, json_template, length_pattern))
            else:
                json_loaded.append(process_single_training_group(training_group, json_template, length_pattern))
                # print(process_single_training_group(training_group, json_template, length_pattern))
    else:
        training_group = simulation_log_df.copy()
        json_loaded.append(process_single_training_group(training_group, json_template, length_pattern))
        # print(process_single_training_group(training_group, json_template, length_pattern))

        
    
    created_json_files = []
    print('Commands at cluster:')

    # return json_loaded    
    for complete_json in json_loaded:
        #put cluster path in the json file
        right_files = []
        for file_combinations in complete_json['parameters']['files']['values']:
            right_files.append({key: os.path.join(cluster_path,'datasets',value) for key, value in file_combinations.items()})
        complete_json['parameters']['files']['values'] = right_files
        # Serialize the data to a JSON formatted str
        json_output = json.dumps(complete_json, indent=4)
        filename = f"{json_prefix}_model_n_event_{complete_json['parameters']['n_events']['value']}_model_n_individual_{complete_json['parameters']['n_individual']['value']}_max_seq_length_{complete_json['parameters']['max_seq_length']['value']}.json"
        # Write the JSON data to a file
        with open(os.path.join(json_folder_path, filename), 'w') as json_file:
            json_file.write(json_output)
        created_json_files.append(filename)
        print(f"python {os.path.join(cluster_path, program_name)} --config_file_path {os.path.join(cluster_path, json_folder_path, filename)} --project_name {project_name}")
    print("Above commands are for the cluster path: ", cluster_path)
    print("Your JSON file has been created in the folder: ", json_folder_path)
    print("All json files created: ", created_json_files)
    
    
    
temp = generate_warmup_json(
    json_prefix = "config_sweep_warmup_",
    program_name = "src/main_run_sweep_without_init_warmup_cosine_annealing.py",
    project_name = "training_inter_event_M_M_1",
    simulation_log_path = "simulation_logs_28_3.pkl",
    json_folder_path = "json_files",
    cluster_path = "/user/dm3766/AI_Queueing/events_queue/modeling/"
)