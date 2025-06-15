import os
import matplotlib.pyplot as plt


# Set plotting style
plt.style.use('default')


# Java setup for Windows
def setup_java():
    java_home = r'C:\Program Files\Java\jdk-17'
    os.environ['JAVA_HOME'] = java_home
    os.environ['PATH'] = f"{java_home}\\bin;" + os.environ.get('PATH', '')
    if 'SPARK_HOME' in os.environ:
        del os.environ['SPARK_HOME']
    print(f"Java configured: {java_home}")
    return True


setup_java()

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql.types import *


def create_spark_session():
    """Create Spark session for fraud analysis"""
    try:
        spark = SparkSession.builder \
            .appName("FraudDetectionAnalysis") \
            .master("local[*]") \
            .config("spark.driver.memory", "4g") \
            .config("spark.driver.maxResultSize", "2g") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()

        spark.sparkContext.setLogLevel("WARN")
        print("Spark session created successfully for full dataset processing")
        return spark

    except Exception as e:
        print(f"Error creating Spark session: {e}")
        raise


def load_and_process_fraud_data(spark, sample_fraction=None):
    """Load fraud data and prepare for analysis"""
    file_path = "fraudTrain.csv"

    df_raw = spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)

    print(f"Available columns: {df_raw.columns}")

    # Select and clean required columns
    df_prepared = df_raw.select(
        col("trans_date_trans_time").alias("transaction_time"),
        col("category").alias("merchant_category"),
        col("amt").cast(DoubleType()).alias("amount"),
        col("first").alias("first_name"),
        col("last").alias("last_name"),
        col("gender").alias("gender"),
        col("city").alias("city"),
        col("state").alias("state"),
        col("lat").cast(DoubleType()).alias("latitude"),
        col("long").cast(DoubleType()).alias("longitude"),
        col("city_pop").cast(IntegerType()).alias("city_population"),
        col("job").alias("job"),
        col("is_fraud").cast(IntegerType()).alias("is_fraud")
    ).filter(
        (col("amount").isNotNull()) &
        (col("merchant_category").isNotNull()) &
        (col("state").isNotNull()) &
        (col("is_fraud").isNotNull()) &
        (col("amount") > 0) &
        (col("amount") < 10000)  # Filter out unrealistic amounts
    )

    # Use full dataset or sample for processing
    if sample_fraction is not None:
        print(f"Using {sample_fraction * 100}% sample for testing...")
        df_final = df_prepared.sample(sample_fraction, seed=42)
    else:
        print("Processing FULL dataset...")
        df_final = df_prepared

    total_records = df_final.count()
    fraud_records = df_final.filter(col("is_fraud") == 1).count()
    fraud_rate = (fraud_records / total_records) * 100 if total_records > 0 else 0

    print(f"Dataset size: {total_records:,} records")
    print(f"Fraud records: {fraud_records:,} ({fraud_rate:.2f}%)")

    return df_final


def analyze_fraud_by_category(df):
    """Analyze fraud rates by merchant category"""
    print("Analyzing fraud patterns by merchant category...")

    category_analysis = df.groupBy("merchant_category").agg(
        count("*").alias("total_transactions"),
        sum("is_fraud").alias("fraud_transactions"),
        avg("amount").alias("avg_amount"),
        (sum("is_fraud") / count("*") * 100).alias("fraud_rate_percent")
    ).filter(
        col("total_transactions") >= 10  # Only categories with meaningful sample size
    ).orderBy(col("fraud_rate_percent").desc())

    return category_analysis


def analyze_fraud_amounts(df):
    """Analyze transaction amounts for fraud vs legitimate"""
    print("Analyzing transaction amount patterns...")

    # Amount analysis by fraud status
    amount_analysis = df.groupBy("is_fraud").agg(
        count("*").alias("transaction_count"),
        avg("amount").alias("avg_amount"),
        expr("percentile_approx(amount, 0.5)").alias("median_amount"),
        min("amount").alias("min_amount"),
        max("amount").alias("max_amount"),
        stddev("amount").alias("stddev_amount")
    )

    # Amount range analysis
    amount_ranges = df.withColumn(
        "amount_range",
        when(col("amount") < 10, "Under $10")
        .when(col("amount") < 50, "$10-$50")
        .when(col("amount") < 100, "$50-$100")
        .when(col("amount") < 500, "$100-$500")
        .otherwise("Over $500")
    ).groupBy("amount_range", "is_fraud").agg(
        count("*").alias("transaction_count")
    ).orderBy("amount_range", "is_fraud")

    return amount_analysis, amount_ranges


def analyze_geographic_patterns(df):
    """Analyze fraud patterns by geographic location"""
    print("Analyzing geographic fraud patterns...")

    state_analysis = df.groupBy("state").agg(
        count("*").alias("total_transactions"),
        sum("is_fraud").alias("fraud_transactions"),
        (sum("is_fraud") / count("*") * 100).alias("fraud_rate_percent"),
        avg("city_population").alias("avg_city_population")
    ).filter(
        col("total_transactions") >= 20  # States with meaningful sample size
    ).orderBy(col("fraud_rate_percent").desc())

    return state_analysis


def analyze_time_patterns(df):
    """Analyze fraud patterns by time"""
    print("Analyzing time-based fraud patterns...")

    # Extract hour from transaction time
    df_with_time = df.withColumn(
        "hour",
        hour(to_timestamp(col("transaction_time"), "dd/MM/yyyy HH:mm"))
    ).filter(col("hour").isNotNull())

    time_analysis = df_with_time.groupBy("hour").agg(
        count("*").alias("total_transactions"),
        sum("is_fraud").alias("fraud_transactions"),
        (sum("is_fraud") / count("*") * 100).alias("fraud_rate_percent")
    ).orderBy("hour")

    return time_analysis


def create_visualizations(category_results, amount_results, geographic_results, time_results):
    """Create and save visualization charts"""
    # Create output directory
    if not os.path.exists('fraud_charts'):
        os.makedirs('fraud_charts')

    # 1. Merchant Category Fraud Rate Chart
    category_data = category_results.limit(10).collect()
    categories = [row['merchant_category'] for row in category_data]
    fraud_rates = [row['fraud_rate_percent'] for row in category_data]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(categories, fraud_rates,
                   color=['#e74c3c' if rate > 1.0 else '#f39c12' if rate > 0.4 else '#27ae60' for rate in fraud_rates])
    plt.title('Top 10 Merchant Categories by Fraud Rate', fontsize=16, fontweight='bold')
    plt.xlabel('Merchant Category', fontsize=12)
    plt.ylabel('Fraud Rate (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Add value labels on bars
    for bar, rate in zip(bars, fraud_rates):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{rate:.2f}%', ha='center', va='bottom', fontweight='bold')

    plt.savefig('fraud_charts/merchant_fraud_rates.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Amount Comparison Chart
    amount_data = amount_results.collect()
    fraud_amounts = [row['avg_amount'] for row in amount_data if row['is_fraud'] == 1]
    legit_amounts = [row['avg_amount'] for row in amount_data if row['is_fraud'] == 0]

    plt.figure(figsize=(8, 6))
    categories_amt = ['Fraudulent', 'Legitimate']
    amounts = [fraud_amounts[0], legit_amounts[0]]
    colors = ['#e74c3c', '#27ae60']

    bars = plt.bar(categories_amt, amounts, color=colors)
    plt.title('Average Transaction Amount: Fraud vs Legitimate', fontsize=16, fontweight='bold')
    plt.ylabel('Average Amount ($)', fontsize=12)

    # Add value labels
    for bar, amount in zip(bars, amounts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 f'${amount:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('fraud_charts/amount_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Time Pattern Chart
    print("Creating time pattern chart...")
    time_data = time_results.collect()
    hours = [row['hour'] for row in time_data]
    time_fraud_rates = [row['fraud_rate_percent'] for row in time_data]

    plt.figure(figsize=(14, 6))
    plt.plot(hours, time_fraud_rates, marker='o', linewidth=3, markersize=6, color='#e74c3c')
    plt.fill_between(hours, time_fraud_rates, alpha=0.3, color='#e74c3c')
    plt.title('Fraud Rate by Hour of Day', fontsize=16, fontweight='bold')
    plt.xlabel('Hour of Day (24-hour format)', fontsize=12)
    plt.ylabel('Fraud Rate (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24))

    # Highlight peak hours
    peak_hours = [22, 23]
    for hour in peak_hours:
        if hour in hours:
            idx = hours.index(hour)
            plt.annotate(f'{time_fraud_rates[idx]:.2f}%',
                         (hour, time_fraud_rates[idx]),
                         textcoords="offset points",
                         xytext=(0, 10), ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('fraud_charts/time_pattern.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Top States Chart
    geo_data = geographic_results.limit(10).collect()
    states = [row['state'] for row in geo_data]
    geo_fraud_rates = [row['fraud_rate_percent'] for row in geo_data]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(states, geo_fraud_rates, color='#3498db')
    plt.title('Top 10 States by Fraud Rate', fontsize=16, fontweight='bold')
    plt.xlabel('State', fontsize=12)
    plt.ylabel('Fraud Rate (%)', fontsize=12)

    # Add value labels
    for bar, rate in zip(bars, geo_fraud_rates):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{rate:.2f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('fraud_charts/states_fraud_rates.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("All charts saved to 'fraud_charts/' directory")


def create_summary_tables(category_results, geographic_results, amount_results):
    """Create and save summary tables"""
    print("Creating summary tables...")

    # Merchant Category Table
    print("Merchant Category Analysis:")
    print("-" * 80)
    print(f"{'Rank':<4} {'Category':<20} {'Fraud Rate':<12} {'Transactions':<12} {'Risk Level':<10}")
    print("-" * 80)

    category_data = category_results.limit(10).collect()
    for i, row in enumerate(category_data, 1):
        rate = row['fraud_rate_percent']
        risk = "HIGH" if rate > 1.0 else "MEDIUM" if rate > 0.4 else "LOW"
        print(f"{i:<4} {row['merchant_category']:<20} {rate:<12.2f} {row['total_transactions']:<12,} {risk:<10}")

    print("\n" + "=" * 60)
    print("Geographic Analysis:")
    print("-" * 60)
    print(f"{'Rank':<4} {'State':<6} {'Fraud Rate':<12} {'Transactions':<12}")
    print("-" * 60)

    geo_data = geographic_results.limit(10).collect()
    for i, row in enumerate(geo_data, 1):
        print(f"{i:<4} {row['state']:<6} {row['fraud_rate_percent']:<12.2f} {row['total_transactions']:<12,}")

    print("\n" + "=" * 50)
    print("Amount Analysis:")
    print("-" * 50)
    amount_data = amount_results.collect()
    for row in amount_data:
        fraud_type = "Fraudulent" if row['is_fraud'] == 1 else "Legitimate"
        print(f"{fraud_type}: Average ${row['avg_amount']:.2f}, Median ${row['median_amount']:.2f}")


def main():
    """Main analysis function"""
    print("PySpark Fraud Detection Analysis")
    print("=" * 50)

    spark = create_spark_session()

    try:
        print("\n[1/5] Loading and processing fraud data...")
        df_dataset = load_and_process_fraud_data(spark)

        print("\n[2/5] Analyzing fraud by merchant category...")
        category_results = analyze_fraud_by_category(df_dataset)

        print("\n[3/5] Analyzing transaction amounts...")
        amount_results, amount_range_results = analyze_fraud_amounts(df_dataset)

        print("\n[4/5] Analyzing geographic patterns...")
        geographic_results = analyze_geographic_patterns(df_dataset)

        print("\n[5/5] Analyzing time patterns...")
        time_results = analyze_time_patterns(df_dataset)

        print("\n[6/6] Creating visualizations and tables...")
        create_visualizations(category_results, amount_results, geographic_results, time_results)
        create_summary_tables(category_results, geographic_results, amount_results)

        # Display results
        print("\n" + "=" * 70)
        print("FRAUD DETECTION ANALYSIS RESULTS")
        print("=" * 70)

        print("\nTOP 10 MERCHANT CATEGORIES BY FRAUD RATE:")
        print("-" * 60)
        top_fraud_categories = category_results.limit(10).collect()
        for i, row in enumerate(top_fraud_categories, 1):
            print(f"{i:2d}. {row['merchant_category']:20} | "
                  f"Fraud Rate: {row['fraud_rate_percent']:5.2f}% | "
                  f"Transactions: {row['total_transactions']:,}")

        print("\nAMOUNT ANALYSIS - FRAUD VS LEGITIMATE:")
        print("-" * 50)
        amount_stats = amount_results.collect()
        for row in amount_stats:
            fraud_status = "FRAUDULENT" if row['is_fraud'] == 1 else "LEGITIMATE"
            print(f"{fraud_status:12} | Avg: ${row['avg_amount']:6.2f} | "
                  f"Median: ${row['median_amount']:6.2f} | Count: {row['transaction_count']:,}")

        print("\nTOP 10 STATES BY FRAUD RATE:")
        print("-" * 55)
        top_fraud_states = geographic_results.limit(10).collect()
        for i, row in enumerate(top_fraud_states, 1):
            print(f"{i:2d}. {row['state']:3} | "
                  f"Fraud Rate: {row['fraud_rate_percent']:5.2f}% | "
                  f"Transactions: {row['total_transactions']:,}")

        print("\nFRAUD PATTERNS BY HOUR OF DAY:")
        print("-" * 40)
        time_patterns = time_results.collect()
        for row in time_patterns:
            if row['total_transactions'] >= 10:
                print(f"Hour {row['hour']:2d}:00 | "
                      f"Fraud Rate: {row['fraud_rate_percent']:5.2f}% | "
                      f"Transactions: {row['total_transactions']:,}")

        print("\n" + "=" * 50)
        print("KEY FINDINGS:")
        print("=" * 50)

        if top_fraud_categories:
            highest_risk_category = top_fraud_categories[0]
            print(f"Highest risk category: {highest_risk_category['merchant_category']}")
            print(f"Fraud rate: {highest_risk_category['fraud_rate_percent']:.2f}%")

        if top_fraud_states:
            highest_risk_state = top_fraud_states[0]
            print(f"Highest risk state: {highest_risk_state['state']}")
            print(f"Fraud rate: {highest_risk_state['fraud_rate_percent']:.2f}%")

        # Find peak fraud hour
        if time_patterns:
            peak_fraud_rate = 0
            peak_hour_info = None
            for row in time_patterns:
                if row['total_transactions'] >= 10 and row['fraud_rate_percent'] > peak_fraud_rate:
                    peak_fraud_rate = row['fraud_rate_percent']
                    peak_hour_info = row

            if peak_hour_info:
                print(f"Peak fraud hour: {peak_hour_info['hour']}:00")
                print(f"Fraud rate: {peak_hour_info['fraud_rate_percent']:.2f}%")

        print(f"\nAnalysis based on {df_dataset.count():,} transaction records")

    except Exception as e:
        print(f"Error during analysis: {e}")
        raise

    finally:
        spark.stop()
        print("\nFraud analysis completed successfully.")


if __name__ == "__main__":
    main()