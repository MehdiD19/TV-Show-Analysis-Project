# Analyse Tv Shows with NLP 

## Project Overview

This project focuses on analyzing a TV show by applying Natural Language Processing (NLP) techniques. The analysis is based on subtitles data and involves two key components:

**Theme Classification:**  Use a Zero-shot classifier to assign and score the themes present in the show based on its dialogues.

**Character Network Analysis:** Named Entity Recognition (NER) is employed to identify and track characters. The relationships between characters are visualized through an interactive network graph with NetworkX and PyViz.

Analysis is displayed on a web interface powered by Gradio.

## Installation and Requirements

### Clone the repo:
 `git clone https://github.com/MehdiD19/TV-Show-Analysis-Project.git`

### Install the requirements
You can install the required packages by running `pip install -r requirements.txt`

### Run the code 
You have to run the file `python gradio_app.py` and provide the path of your subtitles data set as well as a path to store the results

### Note
The load_subtitles fucntion expect a csv file with two columns (Character,Script)
