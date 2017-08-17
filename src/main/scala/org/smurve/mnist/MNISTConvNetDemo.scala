package org.smurve.mnist

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.smurve.mnist.config.MyLocalConfig
import org.nd4s.Implicits._
import org.smurve.nd4s.Conv
import org.smurve.transform.{Affine, Grid}
import org.smurve.nd4s._

import scala.util.Random

object MNISTConvNetDemo extends MNISTTools {
  override protected val config = new MyLocalConfig
  override protected val session: SparkSession = SparkSession.builder().appName("MNist").master("local").getOrCreate()
  override protected val sc: SparkContext = session.sparkContext

  /**
    * @param args no args
    */
  def main(args: Array[String]): Unit = {

    val theta1 = vec (
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

    ).reshape(8,4,3) / 6.0

    val theta2 = vec (
      -1,0,1,1,-1,-1,
      -1,0,-1,-1,1,1,
      -1,0,1,-1,1,-1,
      -1,0,-1,1,-1,1,
      -1,0,1,0,0,-1,
      -1,0,-1,0,0,1,
      -1,0,0,-1,1,0,
      -1,0,0,1,-1,0
    ).reshape(8, 3, 2) / 4.0

    val seed = 123

    val theta3 = (Nd4j.rand(289, 200) - 0.5 ) / 10000
    val theta4 = (Nd4j.rand(201, 10) - 0.5) / 1000

    val conv1 = Conv(theta1, 1, 28, 28,3)
    val max1 = MaxPool(1, 2, 2)
    val conv2 = Conv(theta2, 8, 13, 13,2)
    val max2 = MaxPool(8, 2, 2)
    val relu = ReLU()
    val dense1 = FCL(theta3)
    val dense2 = FCL(theta4)
    val output = Euclidean()

    val convNet = conv1 |:| max1 |:| conv2 |:| relu |:| max2 |:| Sigmoid() |:|
      Flatten(8, 6,6) |:| dense1 |:| relu |:| dense2 |:| output

    println("Reading images from file...")
    val ( img_test, lbl_test ) = super.readFromBinary("test")
    val trainingSet_orig = super.readFromBinary("train")
    println("Done.")

    val (img_train, lbl_train) = shuffle(trainingSet_orig, random = new Random(seed))



    val img = img_test(0, ->).reshape(28, 28)
    println(new Grid(img))

    val maps = convNet.ffwd(img.reshape(1,28,28))

    for ( i <- 0 until 8) {
      val values = maps(i, ->, ->)
      val grid = new Grid(values)
      println(grid)
      println()
    }

    val N_TRAIN = 10000

    val optimizer = new SimpleOptimizer(()=>Affine.identity, new Random(seed))

    val trainingSet = (img_train(0->N_TRAIN, ->), lbl_train(0->N_TRAIN, ->))
    val testSet = (img_test, lbl_test)

    optimizer.train(
      model = convNet, nBatches = 100, parallelism = 4,
      trainingSet = trainingSet, testSet = testSet,
      n_epochs = 10, eta = 1e-4, reportEvery = 100)

  }

}
