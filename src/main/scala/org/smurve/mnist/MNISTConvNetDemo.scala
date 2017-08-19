package org.smurve.mnist

import org.apache.spark.SparkContext
import org.apache.spark.sql.SparkSession
import org.nd4s.Implicits._
import org.smurve.mnist.config.MyLocalConfig
import org.smurve.mnist.models.EfficientConvNet
import org.smurve.nd4s._

import scala.util.Random

object MNISTConvNetDemo extends MNISTTools {
  override protected val config = new MyLocalConfig
  override protected val session: SparkSession = SparkSession.builder().appName("MNist").master("local").getOrCreate()
  override protected val sc: SparkContext = session.sparkContext

  val SEED = 123
  val rnd = new Random()

  /**
    * @param args no args
    */
  def main(args: Array[String]): Unit = {

    /**
      * create a neural network
      */
    val convNet = new EfficientConvNet().model

    /**
      * read training dadta
      */
    val (trainingSet, testSet) = readMNIST(numTrain = 2000, numTest = 100)

    /**
      * checking the intermediate steps
      */
    val img = trainingSet._1(0, ->)
    convNet.setParams(
      ("*:MaxPool", "print.output", 0),
      ("*:Conv", "print.output", 0),
      ("*:Conv", "print.stats", true),
      ("*:Dense", "print.stats", true)
    )


    /**
      * choose an optimizer
      */
    val optimizer = new SimpleOptimizer()


    /**
      * train the network
      */
    optimizer.train(
      model = convNet, nBatches = 10, parallelism = 12,
      trainingSet = trainingSet, testSet = testSet,
      n_epochs = 100, eta = 1e-4, reportEveryAfterBatches = 1)


    /**
      * do some work with it
      */
  }

}
