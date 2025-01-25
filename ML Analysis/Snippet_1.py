import pandas as pd  # Importing the pandas library for data manipulation

# List of CSV file paths to be merged
csv_files = [
    "C:/Users/user/Desktop/pdfs/javaTpoint.csv",  
    "C:/Users/user/Desktop/pdfs/extracted_paragraphs.csv",
    "C:/Users/user/Desktop/pdfs/ml_interview_questions.csv",
    "C:/Users/user/Desktop/pdfs/questions_answers.csv"
]

# Initialize an empty list to store DataFrames
dfs = []

# Loop through each file path in the list
for file in csv_files:
    # Read the current CSV file into a DataFrame
    df = pd.read_csv(file)
    
    # Rename 'text' column to 'data' if it exists
    if 'text' in df.columns:
        df.rename(columns={'text': 'data'}, inplace=True)
    # Rename 'Answer' column to 'data'
    elif 'Answer' in df.columns:
        df.rename(columns={'Answer': 'data'}, inplace=True)
    
    # Append the DataFrame containing only the 'data' column to the list
    dfs.append(df[['data']]) 

# Concatenate all DataFrames in the list into a single DataFrame
merged_df = pd.concat(dfs, ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv("merged_files.csv", index=False)
