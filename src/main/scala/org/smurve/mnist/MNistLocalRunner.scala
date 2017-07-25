package org.smurve.mnist

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession

/**
  * Created by wgiersche on 22/07/17.
  */
object MNistLocalRunner extends MNistRunner ( new HDFSConfig ) {

  protected val session: SparkSession = SparkSession.builder().appName("MNist").master("local").getOrCreate()
  override protected val sc: SparkContext = session.sparkContext
}
