name := "SparkJobs"

version := "1.0"

scalaVersion := "2.11.8"

val nd4jVersion = "0.9.1"



val cluster = sys.props.get("cluster").orElse(Some("false")).get

val localLibs = Seq(
  "org.nd4j" % "nd4j-native-platform" % nd4jVersion,
  "org.apache.spark" %% "spark-core" % "2.2.0",
  "org.apache.spark" %% "spark-streaming" % "2.2.0",
  "org.apache.spark" %% "spark-sql" % "2.2.0",
  "org.apache.hadoop" % "hadoop-client" % "2.7.3"
)

val clusterLibs =   Seq(
  //"org.nd4j" % "nd4j-cuda-8.0" % nd4jVersion  % "provided" classifier "" classifier "linux-x86_64",
  "org.nd4j" % "nd4j-native-platform" % nd4jVersion, //classifier "" classifier "linux-x86_64",
  //"org.bytedeco.javacpp-presets" % "cuda" % "8.0-6.0-1.3" classifier "" classifier "linux-x86_64",
  "org.apache.spark" %% "spark-core" % "2.2.0" % "provided",
  "org.apache.spark" %% "spark-streaming" % "2.2.0" % "provided",
  "org.apache.spark" %% "spark-sql" % "2.2.0" % "provided"
)



val provided = if ( cluster == "false" ) localLibs else clusterLibs



libraryDependencies ++= provided

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % "2.2.6" % "test",
  "org.nd4j" %% "nd4s" % nd4jVersion,
  "com.github.fommil.netlib" % "all" % "1.1.2"
)
