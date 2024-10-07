
import pandas as pd

def load_subtitles(dataset_path, number_of_scripts=10):
    df = pd.read_csv(dataset_path)
    #Get rid of the the 1st column
    df = df.drop(df.columns[0], axis=1)

    # Convert all entries to strings and replace NaN with an empty string
    df = df.fillna("").astype(str)

    #Divide the transcript into a certain number of scripts of equal length
    total_lines = len(df)
    lines_per_script = total_lines // number_of_scripts

    # Initialize lists to store the script numbers and concatenated scripts
    script_numbers = []
    scripts = []

    # Loop through the DataFrame and create the scripts
    for i in range(number_of_scripts):
        start_index = i * lines_per_script
        end_index = (i + 1) * lines_per_script if i < 9 else total_lines
        script = "\n".join(df.iloc[start_index:end_index, 0].tolist())
        script_numbers.append(i + 1)
        scripts.append(script)

    # Create the new DataFrame
    new_df = pd.DataFrame({
        "part": script_numbers,
        "script": scripts
    })

    return new_df