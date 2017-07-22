name := "SparkJobs"

version := "1.0"

scalaVersion := "2.11.8"


val cluster = sys.props.get("cluster").orElse(Some("false")).get

val localLibs = Seq(
  "org.apache.spark" %% "spark-core" % "2.2.0",
  "org.apache.spark" %% "spark-streaming" % "2.2.0",
  "org.apache.spark" %% "spark-sql" % "2.2.0"
)

val clusterLibs =   Seq(
  "org.apache.spark" %% "spark-core" % "2.2.0" % "provided",
  "org.apache.spark" %% "spark-streaming" % "2.2.0" % "provided",
  "org.apache.spark" %% "spark-sql" % "2.2.0" % "provided"
)



val provided = if ( cluster == "false" ) localLibs else clusterLibs



libraryDependencies ++= provided