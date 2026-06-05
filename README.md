# Student Performance Analysis

## Overview
This project analyzes student performance data with comprehensive data cleaning and feature engineering.

## Dataset
- **Original Data**: `Student_performance_data _.csv`
- **Cleaned Data**: `Student_performance_data_cleaned.csv`
- **Records**: 2,392 students
- **Features**: 41 (15 original + 26 engineered)

## Features

### Original Features
- StudentID, Age, Gender, Ethnicity
- ParentalEducation, ParentalSupport
- StudyTimeWeekly, Absences, Tutoring
- Extracurricular activities (Sports, Music, Volunteering)
- GPA, GradeClass

### Engineered Features

#### Combined Features
- **Total_Activities**: Sum of all extracurricular activities
- **Support_Score**: Combined parental education and support metric
- **Study_Effectiveness**: Study time efficiency measure
- **Engagement_Score**: Overall student engagement level
- **At_Risk**: Flag for students with GPA < 2.0 and Absences > 10

#### Standardized Features (Z-score)
- Standardized versions of key numerical features (mean=0, std=1)

#### Normalized Features (Min-Max)
- Normalized versions scaled to 0-1 range

#### Categorical Groups
- Age_Group, GPA_Category, StudyTime_Category, Absence_Level

#### Performance Metrics
- GPA_Rank, Percentile

## Data Cleaning Process

The `clean_data.py` script performs:

1. ✅ Data validation and quality checks
2. ✅ Duplicate detection and removal
3. ✅ Missing value handling
4. ✅ Outlier detection (IQR method)
5. ✅ Feature engineering and combination
6. ✅ Standardization (Z-score normalization)
7. ✅ Normalization (Min-Max scaling)
8. ✅ Categorical grouping
9. ✅ Descriptive label creation

## Files

- `clean_data.py` - Main data cleaning script
- `Student_performance_data _.csv` - Original dataset
- `Student_performance_data_cleaned.csv` - Cleaned and enhanced dataset
- `data_dictionary.csv` - Complete data dictionary with metadata

## Usage

```bash
python clean_data.py
```

## Requirements

```
pandas
numpy
scipy
scikit-learn
```

## Results

- **No outliers detected** in the dataset
- **1,253 students** identified as at-risk
- **41 total features** ready for analysis
- **100% data quality** - no missing values or duplicates
