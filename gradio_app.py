import gradio as gr
from theme_classifier import ThemeClassifier
from character_network import NamedEntityRecognizer, CharacterNetworkGenerator
from character_chatbot import CharacterChatBot
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
load_dotenv()

#Function to create the plot of scores
def create_theme_plot(output_df):
    fig, ax = plt.subplots(figsize=(10, 6))  
    sns.barplot(data=output_df, x='Theme', y='Score', ax=ax, palette='viridis')  
    ax.set_title('Series Themes', fontsize=16)  
    ax.set_xlabel('Theme', fontsize=14)  
    ax.set_ylabel('Score', fontsize=14)  
    plt.xticks(rotation=90)  
    plt.tight_layout() 
    plt.close(fig)  
    return fig

#Function to get themes and return the plot of scores
def get_themes(theme_list_str, subtitles_path, save_path):          
    theme_list = theme_list_str.split(",")
    theme_classifier = ThemeClassifier(theme_list)          #Initialize the ThemeClassifier class
    output_df = theme_classifier.get_themes(subtitles_path, save_path)

    output_df = output_df[theme_list]
    output_df = output_df[theme_list].sum().reset_index()       #Sum the scores of the themes
    output_df.columns = ["Theme", "Score"] 
    
    fig = create_theme_plot(output_df)
    return fig

#Function to get the character network
def get_character_network(subtitles_path,ner_path):
    ner = NamedEntityRecognizer()
    ner_df = ner.get_ners(subtitles_path,ner_path)

    character_network_generator = CharacterNetworkGenerator()
    relationship_df = character_network_generator.generate_character_network(ner_df)
    html = character_network_generator.draw_network_graph(relationship_df)

    return html

#Function to get the chatbot response
def chat_wit_character_chatbot(message, history):
    character_chatbot = CharacterChatBot("Med/ff_Llama-3-8B", 
                                         huggingface_token=os.getenv("huggingface_token"))
    
    output = character_chatbot.chat(message, history)
    output = output['content'].strip()
    return output

def main():
    with gr.Blocks() as iface:
        
        # Theme Classification Section 
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Theme Classification (Zero Shot Classifier)</h1>")
                with gr.Row():
                    with gr.Column():
                        plot = gr.Plot()
                    with gr.Column():
                        theme_list = gr.Textbox(label="Themes")
                        subtitles_path = gr.Textbox(label="Script Path")
                        save_path = gr.Textbox(label="Save Path")
                        get_themes_button = gr.Button("Get Themes")
                        get_themes_button.click(get_themes, inputs=[theme_list, subtitles_path, save_path], outputs=[plot])
        
        # Character Network Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Network (NERs and Graphs)</h1>")
                with gr.Row():
                    with gr.Column():
                        network_html = gr.HTML()
                    with gr.Column():
                        subtitles_path = gr.Textbox(label="Subtutles or Script Path")
                        ner_path = gr.Textbox(label="NERs save path")
                        get_network_graph_button = gr.Button("Get Character Network")
                        get_network_graph_button.click(get_character_network, inputs=[subtitles_path,ner_path], outputs=[network_html])

        # Character Chatbot Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Chatbot</h1>")
                gr.ChatInterface(chat_wit_character_chatbot)
    
    iface.launch(share=True)


if __name__ == "__main__":
    main()