# Credit Card Fraud Detection Analysis

## Project Overview

This project analyzes credit card transaction patterns to identify fraud characteristics using PySpark for big data processing. The analysis examines fraud patterns across multiple features including merchant categories, transaction amounts, geographic distribution, and time of day.

## Method and Approach

The project uses Apache Spark for distributed data processing to handle large-scale transaction data efficiently. The analysis employs:

- **Parallel data processing** with PySpark DataFrames
- **Statistical aggregation** and grouping operations
- **Time-series analysis** for temporal patterns
- **Geographic analysis** for location-based insights
- **Data visualization** for result presentation

## Dataset

- **File**: `fraudTrain.csv`
- **Size**: 1.2M+ transaction records
- **Features**: Transaction details, merchant information, customer demographics, location data, fraud labels

## Code Structure

```
fraud_analysis.py
├── Java setup configuration
├── Spark session initialization
├── Data loading and preprocessing
├── Fraud analysis functions:
│   ├── analyze_fraud_by_category()
│   ├── analyze_fraud_amounts()
│   ├── analyze_geographic_patterns()
│   └── analyze_time_patterns()
├── Visualization generation
└── Summary table generation
```

## Analysis Components

### 1. Merchant Category Analysis
- Calculates fraud rates by merchant category
- Identifies high-risk business types
- Ranks categories by fraud percentage with risk level classification

### 2. Transaction Amount Patterns
- Compares fraudulent vs legitimate transaction amounts
- Statistical analysis including averages, medians, and distributions
- Amount range clustering for pattern identification

### 3. Geographic Distribution
- Analyzes fraud rates by state
- Identifies regional fraud hotspots
- Population-adjusted fraud analysis

### 4. Temporal Analysis
- Examines fraud patterns by hour of day
- Identifies peak fraud times
- 24-hour fraud rate visualization

# PySpark Components Used:
- SparkSession for distributed processing
- DataFrame Operations: groupBy(), agg(), filtering, transformations
- Statistical Functions: percentiles, averages, counts
- Window Functions for ranking and distribution analysis