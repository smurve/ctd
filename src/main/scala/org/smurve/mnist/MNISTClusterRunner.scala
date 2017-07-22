package org.smurve.mnist

import org.apache.spark.sql.SparkSession

/**
  * Created by wgiersche on 22/07/17.
  */
object MNISTClusterRunner extends MNistRunner ( new HDFSConfig ) {

  protected val session: SparkSession = SparkSession.builder().appName("MNIST").getOrCreate()
}
