import pandas as pd

def filter_call_quality(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Ensure correct column names
    df.columns = df.columns.str.strip()
    
    # Group by CallRecordId and filter based on the condition
    def filter_rows(group):
        if len(group) == 2:
            row1, row2 = group.iloc[0], group.iloc[1]
            if row1['StreamQuality'] == row1['OverallCallQuality'] and row2['StreamQuality'] == row2['OverallCallQuality']:
                return group  
            elif row1['StreamQuality'] == row1['OverallCallQuality']:
                return row1.to_frame().T 
            elif row2['StreamQuality'] == row2['OverallCallQuality']:
                return row2.to_frame().T  
        return group  
    
    filtered_df = df.groupby('CallRecordId', group_keys=False).apply(filter_rows)
    
    return filtered_df

# Example usage
file_path = "Call Record Streams with Quality Evaluation and Cost Estimation (1).csv"
filtered_data = filter_call_quality(file_path)
filtered_data.to_csv("Filtered_Call_Records.csv", index=False)
