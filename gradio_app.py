import gradio as gr
from Code.theme_classifier import ThemeClassifier
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def create_theme_plot(output_df):
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
    sns.barplot(data=output_df, x='Theme', y='Score', ax=ax, palette='viridis')  # Use a color palette for aesthetics
    ax.set_title('Series Themes', fontsize=16)  # Set the title of the plot
    ax.set_xlabel('Theme', fontsize=14)  # Set the x-axis label
    ax.set_ylabel('Score', fontsize=14)  # Set the y-axis label
    plt.xticks(rotation=90)  # Rotate x-ticks for better readability
    plt.tight_layout()  # Adjust layout to fit everything nicely
    plt.close(fig)  # Close the figure to prevent it from displaying immediately
    return fig

def get_themes(theme_list_str, subtitles_path, save_path):
    theme_list = theme_list_str.split(",")
    theme_classifier = ThemeClassifier(theme_list)
    output_df = theme_classifier.get_themes(subtitles_path, save_path)

    output_df = output_df[theme_list]
    output_df = output_df[theme_list].sum().reset_index()
    output_df.columns = ["Theme", "Score"] 
    
    fig = create_theme_plot(output_df)
    return fig

def main():
    with gr.Blocks() as iface:
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

    iface.launch(share=True)


if __name__ == "__main__":
    main()
"""
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from Code.theme_classifier import ThemeClassifier
import os


def get_themes(theme_list_str, subtitles_path, save_path):
    try:
        print("Button clicked")
        theme_list = theme_list_str.split(",")
        print(f"Theme list: {theme_list}")
        theme_classifier = ThemeClassifier(theme_list)
        output_df = theme_classifier.get_themes(subtitles_path, save_path)
        print("Themes classified")

        output_df = output_df[theme_list]
        output_df = output_df[theme_list].sum().reset_index()
        output_df.columns = ["Theme", "Score"]
        print(f"Output DataFrame: \n{output_df}")

        fig, ax = plt.subplots()
        output_df.plot(kind='barh', x='Theme', y='Score', ax=ax)
        plt.close(fig)
        return fig
    except Exception as e:
        print(f"Error: {e}")
        return gr.HTML(f"<p style='color:red;'>Error: {e}</p>")


def simple_plot():
    data = {
        "Category": ["A", "B", "C"],
        "Values": [10, 20, 30]
    }
    df = pd.DataFrame(data)
    fig, ax = plt.subplots()
    df.plot(kind='barh', x='Category', y='Values', ax=ax)
    plt.close(fig)
    return fig


def main():
    with gr.Blocks() as iface:
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
                        simple_plot_button = gr.Button("Simple Plot")
                        simple_plot_button.click(simple_plot, outputs=[plot])

    iface.launch(share=True)


if __name__ == "__main__":
    main()


"""