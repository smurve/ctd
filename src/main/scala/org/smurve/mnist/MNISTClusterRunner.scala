package org.smurve.mnist

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession

/**
  * Created by wgiersche on 22/07/17.
  */
object MNISTClusterRunner extends MNistRunner ( new HDFSConfig ) {

  private val cfg = new SparkConf().set("spark.cores.max", "8")

  override protected val session: SparkSession = SparkSession.builder().
    config(cfg).
    appName("MNIST").
    getOrCreate()

  override protected lazy val sc: SparkContext = session.sparkContext

}
