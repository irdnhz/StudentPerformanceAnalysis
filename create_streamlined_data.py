import pandas as pd

# Load the full cleaned data
df = pd.read_csv("Student_performance_data_cleaned.csv")

# Select only the essential columns
streamlined_columns = [
    # Original features
    'StudentID', 'Age', 'Gender', 'Ethnicity',
    'ParentalEducation', 'StudyTimeWeekly', 'Absences',
    'Tutoring', 'ParentalSupport', 'Extracurricular',
    'Sports', 'Music', 'Volunteering', 'GPA', 'GradeClass',

    # Useful labels (readable)
    'Gender_Label', 'Ethnicity_Label', 'ParentalEducation_Label',
    'ParentalSupport_Label', 'GradeClass_Label',

    # Best engineered features (only the most useful ones)
    'Total_Activities',      # Sum of activities
    'Support_Score',          # Combined parental support
    'Engagement_Score',       # Overall engagement
    'At_Risk',               # Risk flag

    # One set of normalized features (choose normalized OR standardized, not both)
    'GPA_Normalized',
    'StudyTimeWeekly_Normalized',
    'Absences_Normalized'
]

# Create streamlined dataset
df_streamlined = df[streamlined_columns].copy()

print(f"Original cleaned data: {len(df.columns)} columns")
print(f"Streamlined data: {len(df_streamlined.columns)} columns")
print(f"\nColumns included:")
for i, col in enumerate(df_streamlined.columns, 1):
    print(f"{i}. {col}")

# Save streamlined version
df_streamlined.to_csv("Student_performance_data_streamlined.csv", index=False)
print(f"\n✅ Streamlined data saved: Student_performance_data_streamlined.csv")
print(f"Size: {len(df_streamlined.columns)} columns, {len(df_streamlined)} rows")
