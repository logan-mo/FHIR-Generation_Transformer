import json
import os
import csv
import re

import re
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import spacy
from datetime import datetime

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

def feature_engineering(string):
    # Initialize an empty dictionary to store the extracted features
    features = {}

    # Extract resourceType
    resource_type_match = re.search(r'\[resourceType\]\s(.*?)\s', string)
    if resource_type_match:
        resource_type = resource_type_match.group(1)
        features['resourceType'] = resource_type

    # Extract id
    id_match = re.search(r'\[id\]\s(.*?)\s', string)
    if id_match:
        id_value = id_match.group(1)
        features['id'] = id_value
    
    references = re.findall(r'\[reference\]\s(.*?)\s', string)
    if references:
        features['references'] = references
    # Extract date/time features
    date_time_values = re.findall(r'\[.*?]\[.*?]\[.*?DateTime\]\s(.*?)\s', string)
    for i, date_time in enumerate(date_time_values):
        feature_name = f'dateTime_{i+1}'
        date_time_obj = datetime.fromisoformat(date_time)
        features[feature_name + '_year'] = date_time_obj.year
        features[feature_name + '_month'] = date_time_obj.month
        features[feature_name + '_day'] = date_time_obj.day
        features[feature_name + '_hour'] = date_time_obj.hour

    # Process text features with spaCy
    text_values = re.findall(r'\[.*?]\[.*?]\[.*?text\]\s(.*?)\s', string)
    for i, text in enumerate(text_values):
        feature_name = f'text_{i+1}'
        # Apply spaCy for tokenization and linguistic feature extraction
        doc = nlp(text)
        tokens = [token.text for token in doc]
        features[feature_name + '_tokens'] = tokens
        features[feature_name + '_num_tokens'] = len(tokens)

        # Extract relevant linguistic features
        # Example: POS tags, dependency labels, named entities
        pos_tags = [token.pos_ for token in doc]
        features[feature_name + '_pos_tags'] = pos_tags
        dependency_labels = [token.dep_ for token in doc]
        features[feature_name + '_dependency_labels'] = dependency_labels
        named_entities = [ent.label_ for ent in doc.ents]
        features[feature_name + '_named_entities'] = named_entities

    # Categorical encoding - One-Hot Encoding
    categorical_labels = re.findall(r'\[.*?]\[.*?]\[.*?]\[(.*?)\]', string)
    for label in categorical_labels:
        # One-hot encode the categorical variables
        one_hot_encoding = pd.get_dummies([label], prefix=label)
        features.update(one_hot_encoding.iloc[0].to_dict())

        # Return the extracted features as a DataFrame
        
    # Extract other labels and perform encoding
    labels = re.findall(r'\[(.*?)\]', string)
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Perform one-hot encoding
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_labels = onehot_encoder.fit_transform(encoded_labels.reshape(-1, 1))

    # Convert the one-hot encoded labels to a DataFrame
    label_columns = [f'label_{i}' for i in range(onehot_labels.shape[1])]
    labels_df = pd.DataFrame(onehot_labels, columns=label_columns)

    # Merge the label DataFrame with the features dictionary
    features.update(labels_df.to_dict(orient='records')[0])

    # Return the extracted features as a DataFrame
    features_df = pd.DataFrame([features])

    return features_df 

def dict_to_string(d, parent_key=''):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}[{k}]" if parent_key else f"[{k}]"
        if isinstance(v, dict):
            items.append(dict_to_string(v, new_key))
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.append(dict_to_string(item, f"{new_key}[{i}]"))
                else:
                    items.append(f"{new_key}[{i}] {item}")
        else:
            items.append(f"{new_key} {v}")
    return " ".join(items).replace('\n', '')

# Define the directory
directory = 'data'

import re

def extract_references(string):
    references = re.findall(r'\[reference\]\s(.*?)\s', string)
    return references

def extract_resource_type(string):
    resource_type_match = re.search(r'\[resourceType\]\s(.*?)\s', string)
    if resource_type_match:
        resource_type = resource_type_match.group(1)
        return resource_type
    else:
        return None

def extract_id(string):
    id_match = re.search(r'\[id\]\s(.*?)\s', string)
    if id_match:
        resource_id = id_match.group(1)
        return resource_id
    else:
        return None

def extract_codings(string):
    matches = re.findall(r'\[system\]\s(.*?)\s.*?\[code\]\s(.*?)\s', string)
    # 'matches' is a list of tuples where the first item is the system and the second is the code
    # transform it into a list of dictionaries for easier use
    codings = [system + '|' + code for system, code in matches]
    return codings

def extract_date_time_values(string):
    date_time_labels = re.findall(r'\[(.*?)\]', string)
    date_time_values = []

    for label in date_time_labels:
        if 'Date' in label or 'Time' in label:
            value_match = re.search(rf'\[{label}\]\s(.*?)\s', string)
            if value_match:
                date_time_values.append(value_match.group(1))
    
    return date_time_values

def extract_nlp_labels(string):
    nlp_labels = re.findall(r'\[text\]\s(.*?)\s', string)
    return nlp_labels

# Open the output file
with open('data/fhir_to_text.csv', 'w', encoding='utf-8', newline='') as out_file:
    writer = csv.writer(out_file)
    
    # Write the header
    writer.writerow(["resourceType", "data"])

    # Loop over all files in the directory
    for filename in os.listdir(directory):

        # Only process .ndjson files
        if filename.startswith('Allergy'):

            # Open the ndjson file
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as in_file:
                features_df = None
                # Process each line in the file
                for line in in_file:
                    resource = json.loads(line)
                    resource_type = resource.get("resourceType", "Unknown")
                    resource_data = dict_to_string(resource)
                    # Example usage
                    string = resource_data

                    resource_type = extract_resource_type(string)
                    resource_id = extract_id(string)
                    codings = extract_codings(string)
                    date_time_values = extract_date_time_values(string)
                    nlp_labels = extract_nlp_labels(string)
                    references = extract_references(string)
                    #print(string)
                    #print(f"Resource Type: {resource_type}")
                    #print(f"Resource ID: {resource_id}")
                    #print(f"Codings: {codings}")
                    #print(f"Date/Time Values: {date_time_values}")
                    #print(f"NLP Labels: {nlp_labels}")
                    #print(f"References: {references}")
                    # Perform feature engineering on the string
                    if features_df == None:
                        features_df = feature_engineering(string)
                    else:
                        features_df = pd.concat([features_df, feature_engineering(string)], axis=0)
                    # Print the extracted features
                    #print(features_df)
                    print('writing features for' + resource_type + ':' + resource_id +'\n')
                    writer.writerow([resource_type, resource_data])
                features_df.to_csv('output/csv/features.csv')



