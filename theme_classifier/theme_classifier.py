from transformers import pipeline
import torch
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd
import nltk
import os 
import sys
import pathlib
folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path,"../"))                    # Ensure we can import modules from parent directory
from utils.data_loader import load_subtitles
nltk.download('punkt')
nltk.download('punkt_tab')

# Initialize the ThemeClassifier class
class ThemeClassifier:
    def __init__(self, theme_list):                  
        self.model_name = "facebook/bart-large-mnli"
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.theme_list = theme_list
        self.theme_classifier = self.load_model(self.device)

# Load the zero-shot-classification model       
    def load_model(self, device):                 
        theme_classifier = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=device
        )

        return theme_classifier
    
# Run the model on a script by dividing it into batches of 20 sentences, and then averaging the scores
    def get_themes_inference(self, script):         
        script_sentences = sent_tokenize(script)

        #Divide the script into batches of 20 sentences
        sentence_batch_size = 20
        script_batches = []
        for index in range(0, len(script_sentences), sentence_batch_size):
            sent = " ".join(script_sentences[index:index + sentence_batch_size])
            script_batches.append(sent)

        #Run the model 
        theme_output = self.theme_classifier(script_batches, self.theme_list, multi_label=True)

        #Wrangle the output
        themes = {}
        for output in theme_output:
            for label, score in zip(output["labels"], output["scores"]):
                if label not in themes:
                    themes[label] = []
                themes[label].append(score)

        #Get the mean of the scores        
        themes = {k: np.mean(np.array(v)) for k, v in themes.items()}

        return themes
    
    def get_themes(self, dataset_path, save_path=None):

        #Read Save Output if  and return it dataframe
        if save_path is not None and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            return df

        #Load the dataset
        df = load_subtitles(dataset_path)

        #Run the Inference
        output_themes= df["script"].apply(self.get_themes_inference)

        #Add the output to the dataframe
        themes_df = pd.DataFrame(output_themes.tolist())    
        df[themes_df.columns] = themes_df

        #Save the output
        if save_path is not None:
            df.to_csv(save_path, index=False)
        
        return df 