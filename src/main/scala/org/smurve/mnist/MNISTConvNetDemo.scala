package org.smurve.mnist

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.smurve.mnist.config.MyLocalConfig
import org.nd4s.Implicits._
import org.smurve.nd4s.Conv
import org.smurve.transform.Grid
import org.smurve.nd4s._

object MNISTConvNetDemo extends MNISTTools {
  override protected val config = new MyLocalConfig
  override protected val session: SparkSession = SparkSession.builder().appName("MNist").master("local").getOrCreate()
  override protected val sc: SparkContext = session.sparkContext

  /**
    * @param args no args
    */
  def main(args: Array[String]): Unit = {

    val theta = vec (
      0,0,0,
      1,1,1,
      0,0,0,
      -1,-1,-1,

      0,0,0,
      -1,-1,-1,
      0,0,0,
      1,1,1,

      0,0,0,
      1,0,-1,
      1,0,-1,
      1,0,-1,

      0,0,0,
      -1,0,1,
      -1,0,1,
      -1,0,1,

      0,0,0,
      1,1,0,
      1,0,-1,
      0,-1,-1,

      0,0,0,
      -1,-1,0,
      -1,0,1,
      0,1,1,

      0,0,0,
      0,1,1,
      -1,0,1,
      -1,-1,0,

      0,0,0,
      0,-1,-1,
      1,0,-1,
      1,1,0

    ).reshape(8,4,3)

    val conv = Conv(theta, 1, 28, 28,3)
    val relu = ReLU()
    val output = Euclidean()

    val convNet = conv |:| relu |:| output

    val ( images, _ ) = super.readFromBinary("test")

    val img = images(0, ->).reshape(28, 28)
    println(new Grid(img))

    val maps = convNet.ffwd(img.reshape(1,28,28))

    for ( i <- 0 until 8) {
      val values = maps(i, 0, ->, ->)
      val grid = new Grid(values)
      println(grid)
      println()

    }
  }
}
