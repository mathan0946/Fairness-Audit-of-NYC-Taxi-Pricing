name := "TaxiFairnessAudit"

version := "1.0"

scalaVersion := "2.13.17"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core"  % "4.1.1" % "provided",
  "org.apache.spark" %% "spark-sql"   % "4.1.1" % "provided",
  "org.apache.spark" %% "spark-mllib" % "4.1.1" % "provided",
  "org.apache.spark" %% "spark-hive"  % "4.1.1" % "provided"
)

// Assembly settings — build a fat JAR for spark-submit
assembly / mainClass := Some("com.taxifairness.Main")
assembly / assemblyMergeStrategy := {
  case PathList("META-INF", _*) => MergeStrategy.discard
  case _                        => MergeStrategy.first
}
