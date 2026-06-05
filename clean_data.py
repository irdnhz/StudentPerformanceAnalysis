import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the data
df = pd.read_csv("Student_performance_data.csv")

print("=== ORIGINAL DATA ===")
print(f"Shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Create a copy for cleaning
df_clean = df.copy()

# 1. Check for and remove duplicate rows
duplicates = df_clean.duplicated().sum()
if duplicates > 0:
    print(f"\n=== REMOVING {duplicates} DUPLICATE ROWS ===")
    df_clean = df_clean.drop_duplicates()

# 2. Check for duplicate StudentIDs
duplicate_ids = df_clean['StudentID'].duplicated().sum()
if duplicate_ids > 0:
    print(f"\n=== WARNING: {duplicate_ids} DUPLICATE STUDENT IDs FOUND ===")
    df_clean = df_clean.drop_duplicates(subset=['StudentID'], keep='first')

# 3. Handle missing values (if any)
if df_clean.isnull().sum().sum() > 0:
    print("\n=== HANDLING MISSING VALUES ===")
    print(df_clean.isnull().sum())
    # For numerical columns, fill with median
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)

# 4. Validate and clean data ranges
print("\n=== VALIDATING DATA RANGES ===")

# Age should be between 15-18
invalid_age = df_clean[(df_clean['Age'] < 15) | (df_clean['Age'] > 18)]
if len(invalid_age) > 0:
    print(f"Warning: {len(invalid_age)} rows with invalid Age values")
    df_clean = df_clean[(df_clean['Age'] >= 15) & (df_clean['Age'] <= 18)]

# GPA should be between 0-4
invalid_gpa = df_clean[(df_clean['GPA'] < 0) | (df_clean['GPA'] > 4)]
if len(invalid_gpa) > 0:
    print(f"Warning: {len(invalid_gpa)} rows with invalid GPA values")
    df_clean = df_clean[(df_clean['GPA'] >= 0) & (df_clean['GPA'] <= 4)]

# Binary columns (Gender, Tutoring, Extracurricular, Sports, Music, Volunteering) should be 0 or 1
binary_cols = ['Gender', 'Tutoring', 'Extracurricular', 'Sports', 'Music', 'Volunteering']
for col in binary_cols:
    invalid = df_clean[~df_clean[col].isin([0, 1])]
    if len(invalid) > 0:
        print(f"Warning: {len(invalid)} rows with invalid {col} values")
        df_clean = df_clean[df_clean[col].isin([0, 1])]

# StudyTimeWeekly should be positive
invalid_study = df_clean[df_clean['StudyTimeWeekly'] < 0]
if len(invalid_study) > 0:
    print(f"Warning: {len(invalid_study)} rows with negative StudyTimeWeekly")
    df_clean = df_clean[df_clean['StudyTimeWeekly'] >= 0]

# Absences should be non-negative
invalid_absences = df_clean[df_clean['Absences'] < 0]
if len(invalid_absences) > 0:
    print(f"Warning: {len(invalid_absences)} rows with negative Absences")
    df_clean = df_clean[df_clean['Absences'] >= 0]

# 5. Add descriptive labels for categorical columns
print("\n=== ADDING DESCRIPTIVE LABELS ===")

# Gender labels
df_clean['Gender_Label'] = df_clean['Gender'].map({0: 'Female', 1: 'Male'})

# Ethnicity labels (assuming standard encoding)
ethnicity_map = {0: 'Caucasian', 1: 'African American', 2: 'Asian', 3: 'Other'}
df_clean['Ethnicity_Label'] = df_clean['Ethnicity'].map(ethnicity_map)

# Parental Education labels
edu_map = {0: 'None', 1: 'High School', 2: 'Some College', 3: 'Bachelor', 4: 'Higher'}
df_clean['ParentalEducation_Label'] = df_clean['ParentalEducation'].map(edu_map)

# Parental Support labels
support_map = {0: 'None', 1: 'Low', 2: 'Moderate', 3: 'High', 4: 'Very High'}
df_clean['ParentalSupport_Label'] = df_clean['ParentalSupport'].map(support_map)

# Grade Class labels
grade_map = {0.0: 'F', 1.0: 'D', 2.0: 'C', 3.0: 'B', 4.0: 'A'}
df_clean['GradeClass_Label'] = df_clean['GradeClass'].map(grade_map)

# 6. OUTLIER DETECTION AND HANDLING
print("\n=== DETECTING OUTLIERS (IQR METHOD) ===")

# Identify numerical columns for outlier detection
numerical_cols = ['Age', 'StudyTimeWeekly', 'Absences', 'GPA']

outlier_summary = {}
for col in numerical_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
    outlier_count = len(outliers)
    outlier_summary[col] = outlier_count

    if outlier_count > 0:
        print(f"{col}: {outlier_count} outliers detected (range: {lower_bound:.2f} to {upper_bound:.2f})")
        # Mark outliers instead of removing them
        df_clean[f'{col}_Outlier'] = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).astype(int)
    else:
        print(f"{col}: No outliers detected")

# 7. COMBINING FEATURES (FEATURE ENGINEERING)
print("\n=== CREATING COMBINED FEATURES ===")

# Total extracurricular activities
df_clean['Total_Activities'] = (df_clean['Extracurricular'] +
                                  df_clean['Sports'] +
                                  df_clean['Music'] +
                                  df_clean['Volunteering'])

# Support score (combination of parental education and support)
df_clean['Support_Score'] = df_clean['ParentalEducation'] + df_clean['ParentalSupport']

# Study effectiveness (study time per absence - inverse relationship)
df_clean['Study_Effectiveness'] = df_clean['StudyTimeWeekly'] / (df_clean['Absences'] + 1)

# Engagement score (activities + tutoring + study time normalized)
df_clean['Engagement_Score'] = (df_clean['Total_Activities'] * 2 +
                                 df_clean['Tutoring'] * 3 +
                                 (df_clean['StudyTimeWeekly'] / 20) * 5)

# Academic risk flag (low GPA + high absences)
df_clean['At_Risk'] = ((df_clean['GPA'] < 2.0) & (df_clean['Absences'] > 10)).astype(int)

print(f"Total_Activities: Sum of all extracurricular activities (range: {df_clean['Total_Activities'].min()}-{df_clean['Total_Activities'].max()})")
print(f"Support_Score: Combined parental support metric (range: {df_clean['Support_Score'].min()}-{df_clean['Support_Score'].max()})")
print(f"Study_Effectiveness: Study time efficiency measure")
print(f"Engagement_Score: Overall student engagement level")
print(f"At_Risk: Students with GPA < 2.0 and Absences > 10 ({df_clean['At_Risk'].sum()} students)")

# 8. STANDARDIZATION AND NORMALIZATION
print("\n=== STANDARDIZING NUMERICAL FEATURES ===")

# Columns to standardize (Z-score normalization: mean=0, std=1)
cols_to_standardize = ['StudyTimeWeekly', 'Absences', 'GPA',
                       'Support_Score', 'Study_Effectiveness', 'Engagement_Score']

scaler = StandardScaler()
for col in cols_to_standardize:
    if col in df_clean.columns:
        df_clean[f'{col}_Standardized'] = scaler.fit_transform(df_clean[[col]])
        print(f"{col}: Standardized (mean={df_clean[f'{col}_Standardized'].mean():.4f}, std={df_clean[f'{col}_Standardized'].std():.4f})")

# Columns to normalize (Min-Max scaling: 0 to 1)
cols_to_normalize = ['Age', 'StudyTimeWeekly', 'Absences', 'GPA']

print("\n=== NORMALIZING FEATURES (0-1 SCALE) ===")
normalizer = MinMaxScaler()
for col in cols_to_normalize:
    if col in df_clean.columns:
        df_clean[f'{col}_Normalized'] = normalizer.fit_transform(df_clean[[col]])
        print(f"{col}: Normalized (min={df_clean[f'{col}_Normalized'].min():.4f}, max={df_clean[f'{col}_Normalized'].max():.4f})")

# 9. NUMBERS WORK - ADDITIONAL CALCULATIONS
print("\n=== ADDITIONAL CALCULATIONS ===")

# Age groups
df_clean['Age_Group'] = pd.cut(df_clean['Age'],
                                bins=[14, 15, 16, 17, 18],
                                labels=['15', '16', '17', '18'])

# GPA categories
df_clean['GPA_Category'] = pd.cut(df_clean['GPA'],
                                   bins=[-0.01, 1.0, 2.0, 3.0, 4.0],
                                   labels=['Poor', 'Below Average', 'Average', 'Excellent'])

# Study time categories
df_clean['StudyTime_Category'] = pd.cut(df_clean['StudyTimeWeekly'],
                                         bins=[0, 5, 10, 15, 20],
                                         labels=['Low', 'Medium', 'High', 'Very High'])

# Absence categories
df_clean['Absence_Level'] = pd.cut(df_clean['Absences'],
                                    bins=[-1, 5, 10, 15, 30],
                                    labels=['Low', 'Moderate', 'High', 'Very High'])

print(f"Created categorical groups: Age_Group, GPA_Category, StudyTime_Category, Absence_Level")

# Performance metrics
df_clean['GPA_Rank'] = df_clean['GPA'].rank(method='average', ascending=False)
df_clean['Percentile'] = df_clean['GPA'].rank(pct=True) * 100

print(f"Added performance rankings: GPA_Rank and Percentile")

# 10. Reset index
df_clean = df_clean.reset_index(drop=True)

# 11. Summary statistics
print("\n=== CLEANED DATA SUMMARY ===")
print(f"Original shape: {df.shape}")
print(f"Cleaned shape: {df_clean.shape}")
print(f"Rows removed: {df.shape[0] - df_clean.shape[0]}")
print(f"Total columns added: {df_clean.shape[1] - df.shape[1]}")
print(f"\nOutlier summary: {outlier_summary}")
print(f"\nNew features created:")
print(f"  - Combined features: 5 (Total_Activities, Support_Score, Study_Effectiveness, Engagement_Score, At_Risk)")
print(f"  - Standardized features: {len(cols_to_standardize)}")
print(f"  - Normalized features: {len(cols_to_normalize)}")
print(f"  - Categorical groupings: 4 (Age_Group, GPA_Category, StudyTime_Category, Absence_Level)")
print(f"  - Performance metrics: 2 (GPA_Rank, Percentile)")
print(f"\nSummary statistics of key features:\n{df_clean[['GPA', 'Total_Activities', 'Support_Score', 'Engagement_Score']].describe()}")

# 12. Save cleaned data
output_file = "Student_performance_data_cleaned.csv"
df_clean.to_csv(output_file, index=False)
print(f"\n=== CLEANED DATA SAVED TO: {output_file} ===")

# 13. Display sample of cleaned data
print(f"\nSample of cleaned data (first 5 rows, selected columns):")
sample_cols = ['StudentID', 'Age', 'GPA', 'Total_Activities', 'Support_Score',
               'Engagement_Score', 'At_Risk', 'GPA_Category', 'GradeClass_Label']
print(df_clean[sample_cols].head())

# 14. Save a data dictionary
print("\n=== CREATING DATA DICTIONARY ===")
data_dict = pd.DataFrame({
    'Column': df_clean.columns,
    'Data_Type': df_clean.dtypes,
    'Non_Null_Count': df_clean.count(),
    'Null_Count': df_clean.isnull().sum(),
    'Unique_Values': df_clean.nunique()
})
data_dict.to_csv("data_dictionary.csv", index=False)
print("Data dictionary saved to: data_dictionary.csv")
