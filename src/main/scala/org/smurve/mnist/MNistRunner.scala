package org.smurve.mnist

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.smurve.nd4s.SimpleOptimizerDemo.seed
import org.smurve.nd4s._
import org.smurve.transform._

import scala.util.Random


abstract class MNistRunner extends MNISTTools {

  trait Params {

    var STORE_AS = "target/tmp/MNIST_DEMO"
    val N_TRAINING = 60000 // max 60000
    val N_TEST = 1000 // max 10000
    val N_EPOCHS = 3
    val eta: Double = 5e-6 // try 1e-2 to 1e-5 by factors of 3

    val nbatches = 50
    val reportEvery = 10

    val parallelism = 12

    val theta1: INDArray = (Nd4j.rand(seed, 785, 200) - .5) / 1000
    val theta2: INDArray = (Nd4j.rand(seed, 201, 10) - .5) / 1000

    def generator(): Affine = Affine.rand(maxAngle = 12, shear_scale_var = .1,
      max_trans_x = 3, random = new Random(seed))
  }

  /**
    *
    * @param args no args
    */
  def main(args: Array[String]): Unit = {

    println("Reading files from disk...")
    val (img_train, lbl_train) = readFromBinary("train")
    val (img_test, lbl_test) = readFromBinary("test")
    println("Done.")

    new Params {

      private val trainingSet = (img_train(0 -> N_TRAINING, ->), lbl_train(0 -> N_TRAINING, ->))
      private val testSet = (img_test(0 -> N_TEST, ->), lbl_test(0 -> N_TEST, ->))

      val network: Layer = Dense(theta1) !! ReLU() !! Dense(theta2) !! Sigmoid() !! Euclidean()

      private val optimizer = new SimpleSGD(generator = generator, random = new Random(seed))

      optimizer.train(
        model = network, nBatches = nbatches,equiv=equiv10,
        parallel = parallelism,
        trainingSet = trainingSet,
        testSet = Some(testSet),
        n_epochs = N_EPOCHS, eta = eta, reportEveryAfterBatches = reportEvery)

      saveModel ( STORE_AS, Map("Theta1"->theta1, "Theta2"->theta2))

      readAndInferModel (STORE_AS, testSet._1(0->10, ->))
    }


  }

  /**
    * just for demo purpose: Read the weights and infer from the given samples, printing the first result
    * @param name the base name of the parameter file
    * @param testSet the subset to infer from
    */
  def readAndInferModel(name: String, testSet: INDArray ): Unit = {
    val weights = readModel ( name, List("Theta1", "Theta2"))
    val theta1_from_file = weights("Theta1")
    val theta2_from_file = weights("Theta2")
    val new_network: Layer = Dense(theta1_from_file) !! ReLU() !! Dense(theta2_from_file) !! Sigmoid() !! Euclidean()
    val res = new_network.ffwd(testSet)
    val sample = testSet(0,->)
    println(new Grid(sample.reshape(28,28)))
    println(res(0,->))
    println("Bye.")
  }




}
