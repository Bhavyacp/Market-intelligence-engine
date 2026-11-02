from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, trim

def create_spark_session():
    return SparkSession.builder \
        .appName("MarketIntelligenceBigData") \
        .master("local[*]") \
        .getOrCreate()

def run_big_data_pipeline(input_json_path: str, output_parquet_path: str):
    spark = create_spark_session()
    print("Starting PySpark Data Ingestion & Transformation Pipeline...")
    
    # Ingest raw unstructured market news JSON stream
    df = spark.read.json(input_json_path)
    
    # Perform distributed transformation & cleaning
    cleaned_df = df \
        .filter(col("text").isNotNull()) \
        .dropDuplicates(["article_id"]) \
        .withColumn("clean_text", trim(regexp_replace(lower(col("text")), "[^a-zA-Z0-9\\s]", "")))
    
    # Save transformed data to optimized Parquet format
    cleaned_df.write.mode("overwrite").parquet(output_parquet_path)
    print(f"PySpark Pipeline Complete! Processed data saved to {output_parquet_path}")
    
    spark.stop()

if __name__ == "__main__":
    run_big_data_pipeline("data/raw_news.json", "data/processed_parquet")