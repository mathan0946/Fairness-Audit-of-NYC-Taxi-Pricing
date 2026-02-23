// ============================================================
// NYC Taxi Fairness Audit  –  Stage 2: Data Enrichment
// ============================================================
// Big Data Frameworks: Apache Spark SQL, Spark broadcast join, HDFS
//
// Run:
//   spark-shell --driver-memory 8g -i scripts/scala/02_DataEnrichment.scala
//
// Input:  output/processed/taxi_cleaned   (Parquet from Stage 1)
//         data/us_income_zipcode.csv       (US Census income data)
// Output: output/processed/taxi_enriched  (Parquet)
//
// This script:
//   1. Loads 46M cleaned taxi records
//   2. Loads US Census median household income by ZIP code
//   3. Maps each pickup GPS coordinate to the nearest NYC ZIP code
//      using an efficient grid-to-zone bounding-box lookup
//   4. Broadcast-joins with census income data
//   5. Assigns income_category (high / medium / low)
//   6. Saves enriched Parquet for Hive + ML pipeline
// ============================================================

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SaveMode}

println("\n" + "=" * 70)
println("  NYC TAXI FAIRNESS AUDIT – DATA ENRICHMENT (Scala / Spark SQL)")
println("  Spark " + spark.version + "  |  Scala " + util.Properties.versionString)
println("=" * 70)

// ──────────────── Configuration ────────────────
val LOCAL_MODE     = true
val TAXI_INPUT     = if (LOCAL_MODE) "output/processed/taxi_cleaned"
                     else "hdfs:///bigdata/processed/taxi_cleaned"
val CENSUS_INPUT   = if (LOCAL_MODE) "data/us_income_zipcode.csv"
                     else "hdfs:///bigdata/census/us_income_zipcode.csv"
val OUTPUT_PATH    = if (LOCAL_MODE) "output/processed/taxi_enriched"
                     else "hdfs:///bigdata/processed/taxi_enriched"

val HIGH_INCOME    = 75000.0
val MEDIUM_INCOME  = 45000.0
val NYC_MEDIAN     = 60000.0   // fallback for unmatched ZIPs

// ──────────────── Step 1: Load cleaned taxi data ────────────────
println("\n[1/6] Loading cleaned taxi data...")
val taxiDF = spark.read.parquet(TAXI_INPUT)
val taxiCount = taxiDF.count()
println(f"   Loaded $taxiCount%,d records")

// ──────────────── Step 2: Load census income data ────────────────
println("\n[2/6] Loading census income data...")
val censusRaw = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv(CENSUS_INPUT)

val censusDF = censusRaw
  .select(
    col("ZIP").cast("string").alias("census_zip"),
    col("`Households Median Income (Dollars)`").cast("double").alias("median_income")
  )
  .filter(col("median_income").isNotNull && col("median_income") > 0)
  // keep only NYC-area ZIP prefixes (10xxx, 11xxx)
  .filter(col("census_zip").startsWith("10") || col("census_zip").startsWith("11"))
  .withColumn("census_zip", lpad(col("census_zip"), 5, "0"))
  // Deduplicate: one income value per ZIP (average if multiple entries)
  .groupBy("census_zip")
  .agg(round(avg("median_income"), 0).alias("median_income"))

val censusCount = censusDF.count()
println(f"   NYC-area ZIP codes with income data: $censusCount%,d")
censusDF.select(
  min("median_income").alias("min"),
  round(avg("median_income"), 0).alias("avg"),
  max("median_income").alias("max")
).show(false)

// ──────────────── Step 3: NYC zone mapping ────────────────
println("\n[3/6] Building NYC geographic zone mapping...")

// Bounding-box zones covering the main NYC areas
// (latMin, latMax, lonMin, lonMax, zip, borough, neighborhood)
case class ZipZone(latMin: Double, latMax: Double,
                   lonMin: Double, lonMax: Double,
                   zip: String, borough: String, neighborhood: String)

val nycZones = Seq(
  // --- MANHATTAN ---
  ZipZone(40.700, 40.715, -74.020, -74.000, "10004", "Manhattan", "Financial District"),
  ZipZone(40.700, 40.715, -74.000, -73.980, "10038", "Manhattan", "City Hall"),
  ZipZone(40.715, 40.725, -74.015, -74.000, "10013", "Manhattan", "Tribeca"),
  ZipZone(40.715, 40.725, -74.000, -73.990, "10012", "Manhattan", "SoHo"),
  ZipZone(40.715, 40.725, -73.990, -73.975, "10002", "Manhattan", "Lower East Side"),
  ZipZone(40.725, 40.740, -74.010, -74.000, "10014", "Manhattan", "West Village"),
  ZipZone(40.725, 40.740, -74.000, -73.990, "10003", "Manhattan", "Greenwich Village"),
  ZipZone(40.725, 40.740, -73.990, -73.975, "10009", "Manhattan", "East Village"),
  ZipZone(40.740, 40.750, -74.005, -73.995, "10011", "Manhattan", "Chelsea"),
  ZipZone(40.740, 40.750, -73.995, -73.980, "10010", "Manhattan", "Gramercy Park"),
  ZipZone(40.740, 40.750, -73.980, -73.970, "10016", "Manhattan", "Murray Hill"),
  ZipZone(40.750, 40.760, -74.000, -73.990, "10001", "Manhattan", "Penn Station"),
  ZipZone(40.750, 40.760, -73.990, -73.980, "10018", "Manhattan", "Garment District"),
  ZipZone(40.750, 40.760, -73.980, -73.970, "10017", "Manhattan", "Grand Central"),
  ZipZone(40.750, 40.760, -73.970, -73.960, "10022", "Manhattan", "Midtown East"),
  ZipZone(40.760, 40.770, -73.995, -73.985, "10036", "Manhattan", "Times Square"),
  ZipZone(40.760, 40.770, -73.985, -73.975, "10019", "Manhattan", "Midtown West"),
  ZipZone(40.760, 40.770, -73.975, -73.965, "10020", "Manhattan", "Rockefeller Center"),
  ZipZone(40.760, 40.770, -73.965, -73.955, "10065", "Manhattan", "Lenox Hill"),
  ZipZone(40.770, 40.785, -73.985, -73.970, "10023", "Manhattan", "Upper West Side"),
  ZipZone(40.770, 40.785, -73.970, -73.955, "10021", "Manhattan", "Upper East Side"),
  ZipZone(40.785, 40.800, -73.980, -73.965, "10024", "Manhattan", "UWS North"),
  ZipZone(40.785, 40.800, -73.965, -73.945, "10028", "Manhattan", "Yorkville"),
  ZipZone(40.800, 40.815, -73.975, -73.958, "10025", "Manhattan", "Morningside Hts"),
  ZipZone(40.800, 40.815, -73.958, -73.940, "10029", "Manhattan", "East Harlem"),
  ZipZone(40.815, 40.830, -73.960, -73.940, "10027", "Manhattan", "Harlem"),
  ZipZone(40.815, 40.830, -73.940, -73.925, "10035", "Manhattan", "East Harlem N"),
  ZipZone(40.830, 40.850, -73.955, -73.935, "10031", "Manhattan", "Hamilton Heights"),
  ZipZone(40.850, 40.870, -73.940, -73.920, "10032", "Manhattan", "Washington Hts"),
  ZipZone(40.870, 40.885, -73.930, -73.910, "10034", "Manhattan", "Inwood"),

  // --- BROOKLYN ---
  ZipZone(40.685, 40.700, -74.000, -73.980, "11201", "Brooklyn", "Brooklyn Heights"),
  ZipZone(40.685, 40.700, -73.980, -73.960, "11205", "Brooklyn", "Fort Greene"),
  ZipZone(40.700, 40.720, -73.965, -73.945, "11211", "Brooklyn", "Williamsburg"),
  ZipZone(40.700, 40.715, -73.945, -73.925, "11206", "Brooklyn", "East Williamsburg"),
  ZipZone(40.720, 40.735, -73.960, -73.940, "11222", "Brooklyn", "Greenpoint"),
  ZipZone(40.690, 40.700, -73.960, -73.940, "11238", "Brooklyn", "Prospect Heights"),
  ZipZone(40.670, 40.690, -73.995, -73.975, "11217", "Brooklyn", "Park Slope"),
  ZipZone(40.655, 40.670, -73.995, -73.975, "11215", "Brooklyn", "South Slope"),
  ZipZone(40.680, 40.695, -73.955, -73.930, "11216", "Brooklyn", "Bed-Stuy"),
  ZipZone(40.665, 40.680, -73.960, -73.935, "11225", "Brooklyn", "Crown Heights"),
  ZipZone(40.645, 40.665, -73.965, -73.945, "11226", "Brooklyn", "Flatbush"),
  ZipZone(40.640, 40.660, -74.020, -73.995, "11220", "Brooklyn", "Sunset Park"),
  ZipZone(40.620, 40.640, -73.990, -73.965, "11218", "Brooklyn", "Kensington"),
  ZipZone(40.660, 40.680, -73.930, -73.905, "11233", "Brooklyn", "Ocean Hill"),
  ZipZone(40.660, 40.680, -73.910, -73.885, "11207", "Brooklyn", "East New York"),
  ZipZone(40.570, 40.600, -73.995, -73.970, "11224", "Brooklyn", "Coney Island"),
  ZipZone(40.630, 40.650, -74.010, -73.985, "11232", "Brooklyn", "Industry City"),
  ZipZone(40.675, 40.695, -74.005, -73.985, "11231", "Brooklyn", "Red Hook"),

  // --- QUEENS ---
  ZipZone(40.740, 40.760, -73.940, -73.915, "11101", "Queens", "Long Island City"),
  ZipZone(40.760, 40.780, -73.930, -73.905, "11106", "Queens", "Astoria"),
  ZipZone(40.740, 40.755, -73.915, -73.895, "11104", "Queens", "Sunnyside"),
  ZipZone(40.740, 40.755, -73.895, -73.870, "11377", "Queens", "Woodside"),
  ZipZone(40.750, 40.770, -73.870, -73.845, "11372", "Queens", "Jackson Heights"),
  ZipZone(40.730, 40.745, -73.890, -73.865, "11373", "Queens", "Elmhurst"),
  ZipZone(40.745, 40.765, -73.845, -73.815, "11368", "Queens", "Corona"),
  ZipZone(40.760, 40.780, -73.835, -73.810, "11354", "Queens", "Flushing"),
  ZipZone(40.715, 40.735, -73.870, -73.845, "11374", "Queens", "Rego Park"),
  ZipZone(40.715, 40.735, -73.850, -73.825, "11375", "Queens", "Forest Hills"),
  ZipZone(40.700, 40.720, -73.810, -73.785, "11432", "Queens", "Jamaica"),
  ZipZone(40.665, 40.685, -73.850, -73.815, "11414", "Queens", "Howard Beach"),
  ZipZone(40.640, 40.665, -73.800, -73.760, "11430", "Queens", "JFK Airport"),

  // --- BRONX ---
  ZipZone(40.810, 40.830, -73.930, -73.905, "10451", "Bronx", "South Bronx"),
  ZipZone(40.820, 40.840, -73.925, -73.900, "10454", "Bronx", "Mott Haven"),
  ZipZone(40.830, 40.855, -73.920, -73.895, "10456", "Bronx", "Morrisania"),
  ZipZone(40.840, 40.860, -73.905, -73.880, "10459", "Bronx", "Longwood"),
  ZipZone(40.855, 40.875, -73.910, -73.885, "10457", "Bronx", "Tremont"),
  ZipZone(40.860, 40.880, -73.895, -73.870, "10460", "Bronx", "West Farms"),
  ZipZone(40.875, 40.895, -73.895, -73.860, "10458", "Bronx", "Fordham"),
  ZipZone(40.840, 40.860, -73.860, -73.830, "10462", "Bronx", "Parkchester"),
  ZipZone(40.880, 40.910, -73.910, -73.890, "10463", "Bronx", "Riverdale"),

  // --- STATEN ISLAND ---
  ZipZone(40.625, 40.650, -74.100, -74.070, "10301", "Staten Island", "St. George"),
  ZipZone(40.580, 40.625, -74.180, -74.130, "10314", "Staten Island", "Willowbrook")
)

println(s"   Defined ${nycZones.size} geographic zones across 5 boroughs")

// ──────────────── Step 4: Map GPS → ZIP/Borough/Neighborhood via UDFs ────────────────
println("\n[4/6] Mapping pickup GPS coordinates to ZIP codes...")

// Spark SQL UDFs (serialised once, executed per-row on executors)
val mapToZip = udf((lat: Double, lon: Double) => {
  if (lat == 0.0 || lon == 0.0) null: String
  else nycZones.find(z =>
    lat >= z.latMin && lat < z.latMax && lon >= z.lonMin && lon < z.lonMax
  ).map(_.zip).orNull
})

val mapToBorough = udf((lat: Double, lon: Double) => {
  if (lat == 0.0 || lon == 0.0) null: String
  else nycZones.find(z =>
    lat >= z.latMin && lat < z.latMax && lon >= z.lonMin && lon < z.lonMax
  ).map(_.borough).orNull
})

val mapToNeighborhood = udf((lat: Double, lon: Double) => {
  if (lat == 0.0 || lon == 0.0) null: String
  else nycZones.find(z =>
    lat >= z.latMin && lat < z.latMax && lon >= z.lonMin && lon < z.lonMax
  ).map(_.neighborhood).orNull
})

val taxiWithGeo = taxiDF
  .withColumn("pickup_zip",    mapToZip(col("pickup_latitude"), col("pickup_longitude")))
  .withColumn("borough",       mapToBorough(col("pickup_latitude"), col("pickup_longitude")))
  .withColumn("neighborhood",  mapToNeighborhood(col("pickup_latitude"), col("pickup_longitude")))
  .filter(col("pickup_zip").isNotNull)   // keep only mapped trips

val mappedCount = taxiWithGeo.count()
val mappedPct = (mappedCount.toDouble / taxiCount * 100)
println(f"   Mapped $mappedCount%,d trips to ZIP codes ($mappedPct%.1f%%)")

println("\n   Borough distribution:")
taxiWithGeo.groupBy("borough")
  .agg(count("*").alias("trips"))
  .orderBy(desc("trips"))
  .show(false)

// ──────────────── Step 5: Broadcast-join with census income ────────────────
println("\n[5/6] Broadcast-joining with census income data...")

val enrichedDF = taxiWithGeo
  .join(broadcast(censusDF), taxiWithGeo("pickup_zip") === censusDF("census_zip"), "left")
  .drop("census_zip")
  // Fallback for unmatched ZIPs
  .withColumn("median_income", coalesce(col("median_income"), lit(NYC_MEDIAN)))
  // Income categories
  .withColumn("income_category",
    when(col("median_income") >= HIGH_INCOME, "high")
    .when(col("median_income") >= MEDIUM_INCOME, "medium")
    .otherwise("low"))

println("   Income category distribution:")
enrichedDF.groupBy("income_category")
  .agg(
    count("*").alias("trip_count"),
    round(avg("fare_amount"), 2).alias("avg_fare"),
    round(avg("trip_distance"), 2).alias("avg_distance"),
    round(avg("median_income"), 0).alias("avg_area_income")
  )
  .orderBy("income_category")
  .show(false)

// ──────────────── Step 6: Save enriched Parquet ────────────────
println(s"\n[6/6] Saving enriched data to: $OUTPUT_PATH")

enrichedDF
  .coalesce(20)   // avoid expensive full shuffle; coalesce reduces partitions in-place
  .write
  .mode(SaveMode.Overwrite)
  .option("compression", "snappy")
  .parquet(OUTPUT_PATH)

val finalCount = enrichedDF.count()

println("\n" + "=" * 70)
println("  DATA ENRICHMENT COMPLETE")
println("=" * 70)
println(f"  Input records:    $taxiCount%,d")
println(f"  Mapped to ZIPs:   $mappedCount%,d ($mappedPct%.1f%%)")
println(f"  Final enriched:   $finalCount%,d")
println(f"  Output:           $OUTPUT_PATH")
println("  New columns:      pickup_zip, borough, neighborhood,")
println("                    median_income, income_category")
println()

println("  Sample enriched data:")
enrichedDF.select(
  "trip_id", "fare_amount", "trip_distance",
  "borough", "neighborhood", "median_income", "income_category"
).show(10, false)

println("=" * 70)
println("  Next: Run 03_MLPipeline.scala")
println("=" * 70)

System.exit(0)
