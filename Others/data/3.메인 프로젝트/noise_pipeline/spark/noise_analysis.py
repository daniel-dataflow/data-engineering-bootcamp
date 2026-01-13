from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, avg
import os
import shutil


RAW_DATA_PATH = "data/raw/gangnam_noise_data.csv"
OUTPUT_PATH = "data/processed/result.csv"

DAY_THRESHOLD = 65
NIGHT_THRESHOLD = 55


def main():
    spark = SparkSession.builder.appName("GangnamNoiseAnalysisCSV").getOrCreate()

    df = spark.read.option("header", "true").option("inferSchema", "true").csv(RAW_DATA_PATH)

    agg_df = df.groupBy("spotCode", "spotName", "spotAddr", "wgs84Lat", "wgs84Lon")\
               .agg(avg("daytimeAve").alias("avg_day_noise"), avg("nightAve").alias("avg_night_noise"))

    result_df = agg_df.withColumn("day_exceeded", when(col("avg_day_noise") > DAY_THRESHOLD, 1).otherwise(0))\
                      .withColumn("night_exceeded", when(col("avg_night_noise") > NIGHT_THRESHOLD, 1).otherwise(0))\
                      .withColumn("need_improvement", when((col("day_exceeded") == 1) | (col("night_exceeded") == 1), 1).otherwise(0))

    result_df.coalesce(1).write.mode("overwrite").option("header", "true").csv("data/processed/result_tmp")

    for file in os.listdir("data/processed/result_tmp"):
        if file.endswith(".csv"):
            shutil.move(f"data/processed/result_tmp/{file}", OUTPUT_PATH)

    shutil.rmtree("data/processed/result_tmp")

    spark.stop()


if __name__ == "__main__":
    main()
