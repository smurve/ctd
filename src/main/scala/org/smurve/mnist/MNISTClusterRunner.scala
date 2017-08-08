package org.smurve.mnist

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.smurve.mnist.config.MyHDFSConfig

/**
  * Created by wgiersche on 22/07/17.
  */
object MNISTClusterRunner extends MNistRunner ( new MyHDFSConfig ) {

  private val cfg = new SparkConf().set("spark.cores.max", "8")

  override protected val session: SparkSession = SparkSession.builder().
    master("spark://daphnis:5050").
    config(cfg).
    appName("MNIST").
    getOrCreate()

  override protected lazy val sc: SparkContext = session.sparkContext

}
