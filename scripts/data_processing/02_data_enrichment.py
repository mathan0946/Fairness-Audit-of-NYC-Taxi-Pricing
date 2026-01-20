"""
============================================
NYC Taxi Fairness Audit - Data Enrichment
============================================
Stage 3B: Enrich taxi data with census income information

This script:
1. Loads cleaned taxi data
2. Loads census income data
3. Joins taxi trips with income data based on location
4. Categorizes trips by income level
5. Saves enriched data for ML training

Author: Big Data Analytics Project
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, lit, round as spark_round,
    avg, count, broadcast, udf
)
from pyspark.sql.types import StringType, DoubleType
import math

# ============================================
# CONFIGURATION
# ============================================

LOCAL_MODE = True

if LOCAL_MODE:
    TAXI_INPUT_PATH = "output/processed/taxi_cleaned"
    CENSUS_INPUT_PATH = "data/us_income_zipcode.csv"
    OUTPUT_PATH = "output/processed/taxi_enriched"
else:
    TAXI_INPUT_PATH = "hdfs:///bigdata/processed/taxi_cleaned"
    CENSUS_INPUT_PATH = "hdfs:///bigdata/census/us_income_zipcode.csv"
    OUTPUT_PATH = "hdfs:///bigdata/processed/taxi_enriched"

# Income category thresholds
HIGH_INCOME_THRESHOLD = 75000
MEDIUM_INCOME_THRESHOLD = 45000

# NYC ZIP code to approximate coordinates mapping (sample)
# In production, use complete geocoding database
NYC_ZIP_COORDINATES = {
    # Manhattan
    "10001": (40.7506, -73.9971, "Chelsea", "Manhattan"),
    "10002": (40.7157, -73.9863, "Lower East Side", "Manhattan"),
    "10003": (40.7317, -73.9893, "East Village", "Manhattan"),
    "10004": (40.6988, -74.0383, "Financial District", "Manhattan"),
    "10005": (40.7069, -74.0089, "Financial District", "Manhattan"),
    "10006": (40.7094, -74.0132, "Financial District", "Manhattan"),
    "10007": (40.7135, -74.0078, "Tribeca", "Manhattan"),
    "10010": (40.7390, -73.9826, "Gramercy Park", "Manhattan"),
    "10011": (40.7418, -74.0002, "Chelsea", "Manhattan"),
    "10012": (40.7258, -73.9981, "SoHo", "Manhattan"),
    "10013": (40.7199, -74.0041, "Tribeca", "Manhattan"),
    "10014": (40.7340, -74.0054, "West Village", "Manhattan"),
    "10016": (40.7459, -73.9783, "Murray Hill", "Manhattan"),
    "10017": (40.7527, -73.9726, "Grand Central", "Manhattan"),
    "10018": (40.7552, -73.9932, "Garment District", "Manhattan"),
    "10019": (40.7654, -73.9860, "Midtown West", "Manhattan"),
    "10020": (40.7587, -73.9787, "Rockefeller Center", "Manhattan"),
    "10021": (40.7690, -73.9590, "Upper East Side", "Manhattan"),
    "10022": (40.7580, -73.9680, "Midtown East", "Manhattan"),
    "10023": (40.7769, -73.9823, "Upper West Side", "Manhattan"),
    "10024": (40.7879, -73.9748, "Upper West Side", "Manhattan"),
    "10025": (40.7990, -73.9668, "Upper West Side", "Manhattan"),
    "10026": (40.8024, -73.9530, "Harlem", "Manhattan"),
    "10027": (40.8116, -73.9530, "Harlem", "Manhattan"),
    "10028": (40.7767, -73.9535, "Upper East Side", "Manhattan"),
    "10029": (40.7919, -73.9441, "East Harlem", "Manhattan"),
    "10030": (40.8183, -73.9431, "Harlem", "Manhattan"),
    "10031": (40.8251, -73.9500, "Hamilton Heights", "Manhattan"),
    "10032": (40.8388, -73.9427, "Washington Heights", "Manhattan"),
    "10033": (40.8497, -73.9349, "Washington Heights", "Manhattan"),
    "10034": (40.8670, -73.9231, "Inwood", "Manhattan"),
    "10035": (40.8007, -73.9352, "East Harlem", "Manhattan"),
    "10036": (40.7590, -73.9897, "Times Square", "Manhattan"),
    "10037": (40.8130, -73.9375, "Harlem", "Manhattan"),
    "10038": (40.7092, -74.0018, "Financial District", "Manhattan"),
    "10039": (40.8264, -73.9361, "Harlem", "Manhattan"),
    "10040": (40.8584, -73.9300, "Washington Heights", "Manhattan"),
    "10044": (40.7617, -73.9500, "Roosevelt Island", "Manhattan"),
    # Bronx
    "10451": (40.8203, -73.9239, "Melrose", "Bronx"),
    "10452": (40.8373, -73.9230, "Highbridge", "Bronx"),
    "10453": (40.8537, -73.9126, "Morris Heights", "Bronx"),
    "10454": (40.8053, -73.9179, "Mott Haven", "Bronx"),
    "10455": (40.8149, -73.9085, "Longwood", "Bronx"),
    "10456": (40.8300, -73.9081, "Morrisania", "Bronx"),
    "10457": (40.8479, -73.8984, "Tremont", "Bronx"),
    "10458": (40.8618, -73.8854, "Belmont", "Bronx"),
    "10459": (40.8253, -73.8930, "Hunts Point", "Bronx"),
    "10460": (40.8427, -73.8792, "West Farms", "Bronx"),
    "10461": (40.8459, -73.8417, "Morris Park", "Bronx"),
    "10462": (40.8431, -73.8573, "Parkchester", "Bronx"),
    "10463": (40.8795, -73.9062, "Kingsbridge", "Bronx"),
    "10464": (40.8677, -73.7967, "City Island", "Bronx"),
    "10465": (40.8227, -73.8218, "Throggs Neck", "Bronx"),
    "10466": (40.8905, -73.8466, "Wakefield", "Bronx"),
    "10467": (40.8738, -73.8712, "Norwood", "Bronx"),
    "10468": (40.8670, -73.8996, "Fordham", "Bronx"),
    "10469": (40.8687, -73.8508, "Eastchester", "Bronx"),
    "10470": (40.8994, -73.8640, "Woodlawn", "Bronx"),
    "10471": (40.9007, -73.8978, "Riverdale", "Bronx"),
    "10472": (40.8294, -73.8689, "Soundview", "Bronx"),
    "10473": (40.8189, -73.8579, "Castle Hill", "Bronx"),
    "10474": (40.8122, -73.8850, "Hunts Point", "Bronx"),
    "10475": (40.8774, -73.8251, "Co-op City", "Bronx"),
    # Brooklyn
    "11201": (40.6934, -73.9899, "Brooklyn Heights", "Brooklyn"),
    "11203": (40.6492, -73.9348, "East Flatbush", "Brooklyn"),
    "11204": (40.6189, -73.9847, "Bensonhurst", "Brooklyn"),
    "11205": (40.6948, -73.9656, "Fort Greene", "Brooklyn"),
    "11206": (40.7014, -73.9428, "Williamsburg", "Brooklyn"),
    "11207": (40.6716, -73.8936, "East New York", "Brooklyn"),
    "11208": (40.6693, -73.8713, "East New York", "Brooklyn"),
    "11209": (40.6223, -74.0305, "Bay Ridge", "Brooklyn"),
    "11210": (40.6281, -73.9470, "Flatlands", "Brooklyn"),
    "11211": (40.7121, -73.9545, "Williamsburg", "Brooklyn"),
    "11212": (40.6628, -73.9127, "Brownsville", "Brooklyn"),
    "11213": (40.6709, -73.9359, "Crown Heights", "Brooklyn"),
    "11214": (40.5996, -73.9965, "Bensonhurst", "Brooklyn"),
    "11215": (40.6626, -73.9862, "Park Slope", "Brooklyn"),
    "11216": (40.6804, -73.9493, "Bedford-Stuyvesant", "Brooklyn"),
    "11217": (40.6820, -73.9798, "Boerum Hill", "Brooklyn"),
    "11218": (40.6433, -73.9774, "Kensington", "Brooklyn"),
    "11219": (40.6325, -73.9968, "Borough Park", "Brooklyn"),
    "11220": (40.6414, -74.0163, "Sunset Park", "Brooklyn"),
    "11221": (40.6920, -73.9276, "Bushwick", "Brooklyn"),
    "11222": (40.7277, -73.9481, "Greenpoint", "Brooklyn"),
    "11223": (40.5970, -73.9733, "Gravesend", "Brooklyn"),
    "11224": (40.5770, -73.9880, "Coney Island", "Brooklyn"),
    "11225": (40.6628, -73.9545, "Crown Heights", "Brooklyn"),
    "11226": (40.6462, -73.9568, "Flatbush", "Brooklyn"),
    "11228": (40.6169, -74.0133, "Dyker Heights", "Brooklyn"),
    "11229": (40.6009, -73.9440, "Sheepshead Bay", "Brooklyn"),
    "11230": (40.6217, -73.9654, "Midwood", "Brooklyn"),
    "11231": (40.6787, -74.0003, "Carroll Gardens", "Brooklyn"),
    "11232": (40.6581, -74.0035, "Sunset Park", "Brooklyn"),
    "11233": (40.6782, -73.9199, "Bedford-Stuyvesant", "Brooklyn"),
    "11234": (40.6046, -73.9117, "Canarsie", "Brooklyn"),
    "11235": (40.5844, -73.9486, "Brighton Beach", "Brooklyn"),
    "11236": (40.6394, -73.9010, "Canarsie", "Brooklyn"),
    "11237": (40.7043, -73.9211, "Bushwick", "Brooklyn"),
    "11238": (40.6792, -73.9635, "Prospect Heights", "Brooklyn"),
    "11239": (40.6475, -73.8793, "East New York", "Brooklyn"),
    # Queens
    "11101": (40.7478, -73.9394, "Long Island City", "Queens"),
    "11102": (40.7715, -73.9260, "Astoria", "Queens"),
    "11103": (40.7627, -73.9127, "Astoria", "Queens"),
    "11104": (40.7442, -73.9202, "Sunnyside", "Queens"),
    "11105": (40.7787, -73.9063, "Astoria", "Queens"),
    "11106": (40.7614, -73.9314, "Astoria", "Queens"),
    "11354": (40.7683, -73.8276, "Flushing", "Queens"),
    "11355": (40.7511, -73.8215, "Flushing", "Queens"),
    "11356": (40.7848, -73.8418, "College Point", "Queens"),
    "11357": (40.7861, -73.8113, "Whitestone", "Queens"),
    "11358": (40.7604, -73.7959, "Flushing", "Queens"),
    "11360": (40.7815, -73.7810, "Bayside", "Queens"),
    "11361": (40.7639, -73.7712, "Bayside", "Queens"),
    "11362": (40.7565, -73.7353, "Little Neck", "Queens"),
    "11363": (40.7729, -73.7462, "Douglaston", "Queens"),
    "11364": (40.7454, -73.7598, "Oakland Gardens", "Queens"),
    "11365": (40.7395, -73.7950, "Fresh Meadows", "Queens"),
    "11366": (40.7266, -73.7867, "Fresh Meadows", "Queens"),
    "11367": (40.7300, -73.8237, "Kew Gardens Hills", "Queens"),
    "11368": (40.7497, -73.8527, "Corona", "Queens"),
    "11369": (40.7630, -73.8726, "East Elmhurst", "Queens"),
    "11370": (40.7656, -73.8930, "Jackson Heights", "Queens"),
    "11372": (40.7518, -73.8834, "Jackson Heights", "Queens"),
    "11373": (40.7388, -73.8786, "Elmhurst", "Queens"),
    "11374": (40.7264, -73.8615, "Rego Park", "Queens"),
    "11375": (40.7210, -73.8454, "Forest Hills", "Queens"),
    "11377": (40.7449, -73.9067, "Woodside", "Queens"),
    "11378": (40.7243, -73.9097, "Maspeth", "Queens"),
    "11379": (40.7168, -73.8795, "Middle Village", "Queens"),
    "11385": (40.7003, -73.8898, "Ridgewood", "Queens"),
    "11411": (40.6919, -73.7363, "Cambria Heights", "Queens"),
    "11412": (40.6985, -73.7590, "St. Albans", "Queens"),
    "11413": (40.6715, -73.7522, "Springfield Gardens", "Queens"),
    "11414": (40.6577, -73.8442, "Howard Beach", "Queens"),
    "11415": (40.7082, -73.8288, "Kew Gardens", "Queens"),
    "11416": (40.6847, -73.8507, "Ozone Park", "Queens"),
    "11417": (40.6764, -73.8439, "Ozone Park", "Queens"),
    "11418": (40.6995, -73.8340, "Richmond Hill", "Queens"),
    "11419": (40.6879, -73.8229, "South Richmond Hill", "Queens"),
    "11420": (40.6738, -73.8177, "South Ozone Park", "Queens"),
    "11421": (40.6931, -73.8587, "Woodhaven", "Queens"),
    "11422": (40.6604, -73.7354, "Rosedale", "Queens"),
    "11423": (40.7156, -73.7681, "Hollis", "Queens"),
    "11426": (40.7361, -73.7218, "Bellerose", "Queens"),
    "11427": (40.7309, -73.7459, "Queens Village", "Queens"),
    "11428": (40.7208, -73.7422, "Queens Village", "Queens"),
    "11429": (40.7099, -73.7389, "Queens Village", "Queens"),
    "11430": (40.6462, -73.7852, "JFK Airport", "Queens"),
    "11432": (40.7159, -73.7929, "Jamaica", "Queens"),
    "11433": (40.6985, -73.7861, "Jamaica", "Queens"),
    "11434": (40.6768, -73.7758, "Jamaica", "Queens"),
    "11435": (40.7014, -73.8092, "Jamaica", "Queens"),
    "11436": (40.6764, -73.7966, "Jamaica", "Queens"),
    # Staten Island
    "10301": (40.6405, -74.0902, "St. George", "Staten Island"),
    "10302": (40.6310, -74.1378, "Port Richmond", "Staten Island"),
    "10303": (40.6321, -74.1679, "Mariners Harbor", "Staten Island"),
    "10304": (40.6037, -74.0932, "Stapleton", "Staten Island"),
    "10305": (40.5964, -74.0752, "Rosebank", "Staten Island"),
    "10306": (40.5672, -74.1111, "New Dorp", "Staten Island"),
    "10307": (40.5093, -74.2419, "Tottenville", "Staten Island"),
    "10308": (40.5518, -74.1493, "Great Kills", "Staten Island"),
    "10309": (40.5298, -74.2191, "Charleston", "Staten Island"),
    "10310": (40.6326, -74.1167, "West Brighton", "Staten Island"),
    "10312": (40.5442, -74.1798, "Annadale", "Staten Island"),
    "10314": (40.5883, -74.1640, "New Springville", "Staten Island"),
}


# ============================================
# SPARK SESSION
# ============================================

def create_spark_session():
    """Create and configure Spark session."""
    spark = SparkSession.builder \
        .appName("NYC_Taxi_Fairness_DataEnrichment") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ============================================
# HELPER FUNCTIONS
# ============================================

def find_nearest_zip(lat, lon):
    """Find the nearest ZIP code based on coordinates."""
    if lat is None or lon is None:
        return None
    
    min_distance = float('inf')
    nearest_zip = None
    
    for zip_code, (zip_lat, zip_lon, _, _) in NYC_ZIP_COORDINATES.items():
        # Simple Euclidean distance (sufficient for nearby points)
        distance = math.sqrt((lat - zip_lat)**2 + (lon - zip_lon)**2)
        if distance < min_distance:
            min_distance = distance
            nearest_zip = zip_code
    
    return nearest_zip if min_distance < 0.05 else None  # ~3.5 miles threshold


def get_neighborhood(zip_code):
    """Get neighborhood name from ZIP code."""
    if zip_code in NYC_ZIP_COORDINATES:
        return NYC_ZIP_COORDINATES[zip_code][2]
    return "Unknown"


def get_borough(zip_code):
    """Get borough name from ZIP code."""
    if zip_code in NYC_ZIP_COORDINATES:
        return NYC_ZIP_COORDINATES[zip_code][3]
    return "Unknown"


# ============================================
# DATA LOADING
# ============================================

def load_cleaned_taxi_data(spark, input_path):
    """Load cleaned taxi data from Parquet."""
    print(f"\n[1/6] Loading cleaned taxi data from: {input_path}")
    
    df = spark.read.parquet(input_path)
    record_count = df.count()
    print(f"   Loaded {record_count:,} taxi records")
    
    return df


def load_census_data(spark, input_path):
    """Load and process census income data."""
    print(f"\n[2/6] Loading census income data from: {input_path}")
    
    df = spark.read.csv(
        input_path,
        header=True,
        inferSchema=True
    )
    
    # Select and rename relevant columns
    # Based on actual column names in the file
    census_df = df.select(
        col("ZIP").alias("zip_code"),
        col("Geographic Area Name").alias("area_name"),
        col("Households Median Income (Dollars)").alias("median_income")
    ).filter(col("median_income").isNotNull())
    
    # Clean ZIP code format
    census_df = census_df.withColumn(
        "zip_code",
        col("zip_code").cast("string")
    )
    
    # Pad ZIP codes to 5 digits
    census_df = census_df.withColumn(
        "zip_code",
        when(col("zip_code").isNotNull(),
             col("zip_code")).otherwise(None)
    )
    
    record_count = census_df.count()
    print(f"   Loaded {record_count:,} ZIP code income records")
    
    return census_df


# ============================================
# ENRICHMENT FUNCTIONS
# ============================================

def assign_zip_codes_udf(df):
    """Assign ZIP codes to taxi pickups using UDF."""
    print("\n[3/6] Assigning ZIP codes to pickup locations...")
    
    # Register UDF
    find_zip_udf = udf(find_nearest_zip, StringType())
    
    # Apply UDF to find nearest ZIP
    df = df.withColumn(
        "pickup_zip",
        find_zip_udf(col("pickup_latitude"), col("pickup_longitude"))
    )
    
    # Count successful assignments
    assigned_count = df.filter(col("pickup_zip").isNotNull()).count()
    total_count = df.count()
    print(f"   Assigned ZIP codes to {assigned_count:,} / {total_count:,} records ({assigned_count/total_count*100:.1f}%)")
    
    return df


def add_geographic_info(df):
    """Add neighborhood and borough information."""
    print("\n[4/6] Adding neighborhood and borough information...")
    
    # Register UDFs
    neighborhood_udf = udf(get_neighborhood, StringType())
    borough_udf = udf(get_borough, StringType())
    
    df = df.withColumn("neighborhood", neighborhood_udf(col("pickup_zip")))
    df = df.withColumn("borough", borough_udf(col("pickup_zip")))
    
    # Show distribution
    print("   Borough distribution:")
    df.groupBy("borough").count().orderBy("count", ascending=False).show()
    
    return df


def join_with_income(df, census_df):
    """Join taxi data with census income data."""
    print("\n[5/6] Joining with census income data...")
    
    # Broadcast smaller census table for efficient join
    df_enriched = df.join(
        broadcast(census_df),
        df.pickup_zip == census_df.zip_code,
        "left"
    ).drop("zip_code", "area_name")
    
    # Fill missing income with median NYC income
    nyc_median_income = 60000  # Approximate NYC median
    df_enriched = df_enriched.fillna({"median_income": nyc_median_income})
    
    # Add income category
    df_enriched = df_enriched.withColumn(
        "income_category",
        when(col("median_income") >= HIGH_INCOME_THRESHOLD, "high")
        .when(col("median_income") >= MEDIUM_INCOME_THRESHOLD, "medium")
        .otherwise("low")
    )
    
    # Show income category distribution
    print("   Income category distribution:")
    df_enriched.groupBy("income_category").count().orderBy("count", ascending=False).show()
    
    return df_enriched


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main execution function."""
    print("=" * 60)
    print("NYC TAXI FAIRNESS AUDIT - DATA ENRICHMENT")
    print("=" * 60)
    
    spark = create_spark_session()
    
    try:
        # Load data
        taxi_df = load_cleaned_taxi_data(spark, TAXI_INPUT_PATH)
        census_df = load_census_data(spark, CENSUS_INPUT_PATH)
        
        # Enrich data
        taxi_df = assign_zip_codes_udf(taxi_df)
        taxi_df = add_geographic_info(taxi_df)
        enriched_df = join_with_income(taxi_df, census_df)
        
        # Filter out records without valid ZIP codes
        enriched_df = enriched_df.filter(col("pickup_zip").isNotNull())
        
        # Save enriched data
        print(f"\n[6/6] Saving enriched data to: {OUTPUT_PATH}")
        
        enriched_df.write \
            .mode("overwrite") \
            .parquet(OUTPUT_PATH)
        
        final_count = enriched_df.count()
        
        print("\n" + "=" * 60)
        print("DATA ENRICHMENT COMPLETE")
        print("=" * 60)
        print(f"Enriched records:  {final_count:,}")
        print(f"Output location:   {OUTPUT_PATH}")
        print("=" * 60)
        
        # Show sample
        print("\nSample of enriched data:")
        enriched_df.select(
            "trip_id", "trip_distance", "fare_amount",
            "pickup_zip", "neighborhood", "borough",
            "median_income", "income_category"
        ).show(10)
        
        # Summary statistics by income category
        print("\nSummary by income category:")
        enriched_df.groupBy("income_category").agg(
            count("*").alias("trip_count"),
            spark_round(avg("trip_distance"), 2).alias("avg_distance"),
            spark_round(avg("fare_amount"), 2).alias("avg_fare"),
            spark_round(avg("fare_per_mile"), 2).alias("avg_fare_per_mile")
        ).orderBy("income_category").show()
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
