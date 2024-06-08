import pandas as pd

def clean_tsv(input_tsv_path, output_tsv_path):
    # Read the TSV file into a DataFrame
    df = pd.read_csv(input_tsv_path, sep='\t')
    
    # Drop the columns 'label1' to 'label5'
    df = df.drop(columns=['label1', 'label2', 'label3', 'label4', 'label5'])
    
    # Add an index column
    df.insert(0, 'index', range(1, len(df) + 1))
    
    # Reorder the columns
    df = df[['index', 'captionID', 'pairID', 'sentence1_binary_parse', 'sentence2_binary_parse',
             'sentence1_parse', 'sentence2_parse', 'sentence1', 'sentence2', 'gold_label']]
    
    # Save the cleaned DataFrame to a new TSV file
    df.to_csv(output_tsv_path, sep='\t', index=False)

# Usage
input_tsv_path = 'snli_1.0/test.tsv'  # Path to your input TSV file
output_tsv_path = 'snli_1.0/clean/test.tsv'  # Path where you want to save the cleaned TSV file
clean_tsv(input_tsv_path, output_tsv_path)
