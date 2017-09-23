package org.smurve.mnist

import org.smurve.mnist.models.EfficientConvNet
import org.smurve.nd4s._

import scala.util.Random

object MNISTConvNetDemo extends MNISTTools {

  val SEED = 123
  val rnd = new Random()

  /**
    * @param args no args
    */
  def main(args: Array[String]): Unit = {

    println("=============================================================================================")
    println("                      M N I S T   Convolutional Network Demo")
    println("=============================================================================================")

    /**
      * create a neural network
      */
    val convNet = new EfficientConvNet().model

    /**
      * read training dadta
      */
    val (trainingSet, testSet) = readMNIST(numTrain = 2000, numTest = 100)

    /**
      * Parameters
      */
    convNet.setParams(
      ("*:MaxPool", "print.output", 0),
      ("*:Conv", "print.output", 0),
      ("*:Conv", "print.stats", true),
      ("*:Dense", "print.stats", true)
    )


    /**
      * choose an optimizer
      */
    val optimizer = new SimpleSGD()


    /**
      * train the network
      */
    optimizer.train(
      model = convNet, nBatches = 10, parallel = 12, equiv = equiv10,
      trainingSet = trainingSet, testSet = testSet,
      n_epochs = 100, eta = 1e-4, reportEveryAfterBatches = 1)


    /**
      * do some work with it
      */
  }

}
