package org.smurve

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

/**
  * Created by wgiersche on 22/07/17.
  */
object CountingBrown {

  private val node = "daphnis"
  private val home = "/users/wgiersche"
  private val prefix = s"hdfs://$node/$home/"

  def hdfs ( name: String ): String = prefix + name

  def main(args: Array[String]): Unit = {
    val session = SparkSession.builder().appName("WordCount").getOrCreate()
    val sc = session.sparkContext

    println("Starting...")

    count_the_Brown_corpus(sc)

    println("Done.")
    session.stop()
  }

  private def count_the_Brown_corpus(sc: SparkContext) = {
    val file = sc.textFile(hdfs("brown/*"))
    val labeled = file.flatMap(_.split(" ")).map(_.trim).flatMap(
      w => if (w contains "/")
        Some(w.split("/")(0))
      else
        None
    )
    val cnt = labeled.count
    println(s"Counted $cnt")
  }
}
