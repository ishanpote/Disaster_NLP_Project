import pandas as pd

# 1. Load the human-readable processed CSVs
# Using '../' to step out of the 'src' folder and into the root directory
print("Loading processed datasets...")
humaid_df = pd.read_csv(r'F:\Ishan E\3rd_year\6SEM\Project\Disaster_NLP_Project\data\processed\humaid_cleaned.csv')
kaggle_df = pd.read_csv(r'F:\Ishan E\3rd_year\6SEM\Project\Disaster_NLP_Project\data\processed\kaggle_cleaned.csv')
misclassified_df = pd.read_csv(r'F:\Ishan E\3rd_year\6SEM\Project\Disaster_NLP_Project\data\processed\misclassified_tweets.csv')

# 2. Define your conference output file name with your exact User ID
# Saving it with '../' will place the final Excel file cleanly in your root folder
output_filename = r'F:/Ishan E/3rd_year/6SEM/Project/Disaster_NLP_Project/data/processed/NCMLAI1855.xlsx'

# 3. Write them to a single Excel workbook with clearly labeled sheets
print("Converting to MS Excel format...")
with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
    # Main datasets
    humaid_df.to_excel(writer, sheet_name='HumAID Cleaned Data', index=False)
    kaggle_df.to_excel(writer, sheet_name='Kaggle Cleaned Data', index=False)
    
    # LIME diagnostics
    misclassified_df.to_excel(writer, sheet_name='LIME Diagnostic Misses', index=False)

print(f"Success! Dataset compiled and saved as {output_filename} in your project root.")