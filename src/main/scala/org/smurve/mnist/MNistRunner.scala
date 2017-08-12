package org.smurve.mnist

import java.util.UUID

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.smurve.mnist.config.MNistConfig
import org.smurve.nd4s.SimpleOptimizerDemo.seed
import org.smurve.nd4s._
import org.smurve.transform._

import scala.util.Random


abstract class MNistRunner(protected val config: MNistConfig) extends MNISTTools {

  protected val session: SparkSession
  protected val sc: SparkContext

  trait Params {

    var STORE_AS = "target/tmp/MNIST_DEMO"
    val N_TRAINING = 6000 // max 60000
    val N_TEST = 1000 // max 10000
    val N_EPOCHS = 3
    val eta: Double = 1e-4 // try 1e-2 to 1e-5 by factors of 3

    val nbatches = 600
    val reportEvery = 10

    val parallelism = 4

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

      val network: Layer = FCL(theta1) |:| ReLU() |:| FCL(theta2) |:| Sigmoid() |:| Euclidean()

      private val optimizer = new SimpleOptimizer(generator = generator, random = new Random(seed))

      optimizer.train(
        model = network, nBatches = nbatches,
        parallelism = parallelism,
        trainingSet = trainingSet,
        testSet = testSet,
        n_epochs = N_EPOCHS, eta = eta, reportEvery = reportEvery)

      saveModel ( STORE_AS, Map("Theta1"->theta1, "Theta2"->theta2))

      readAndInferModel (STORE_AS, theta1, theta2, testSet._1(0->10, ->))
    }


    session.stop()
  }

  /**
    * just for demo purpose: Read the weights and infer from the given samples, printing the first result
    * @param name the base name of the parameter file
    * @param theta1 the first FCL's weights
    * @param theta2 the second FCL's weights
    * @param testSet the subset to infer from
    */
  def readAndInferModel(name: String, theta1: INDArray, theta2: INDArray, testSet: INDArray ): Unit = {
    val weights = readModel ( name, List("Theta1", "Theta2"))
    val theta1_from_file = weights("Theta1")
    val theta2_from_file = weights("Theta2")
    val new_network: Layer = FCL(theta1) |:| ReLU() |:| FCL(theta2) |:| Sigmoid() |:| Euclidean()
    val res = new_network.ffwd(testSet)
    val sample = testSet(0,->)
    println(new Grid(sample.reshape(28,28)))
    println(res(0,->))
    println("Bye.")
  }


  /**
    * Read the data files
    * @param key either "train" or "test"
    * @return a pair of INDArrays containing images and labels
    */
  def readFromBinary(key: String): (INDArray, INDArray) = {
    val images: MNISTImages = createImagesFromBinary(s"input/$key").first()
    val labels = createLabelsFromBinary(s"input/$key-labels").first()

    (images.asINDarray, labels.asINDarray)
  }



  /**
    * resolve in local fs or hdfs, depending on config
    *
    * @param name short name of the file
    * @return
    */
  protected def resolve(name: String): String = config.prefix + name


  /**
    * @param name name of file or directory to read from
    * @return an RDD of MNISTImages wrappers
    */
  protected def createImagesFromBinary(name: String): RDD[MNISTImages] = {

    val rawImages = sc.binaryFiles(resolve(name))

    rawImages.map(p => {
      val stream = p._2
      new MNISTImages(stream.toArray())
    })
  }

  /**
    * @param name name of file or directory to read from
    * @return an RDD of MNISTLabel wrappers
    */
  protected def createLabelsFromBinary(name: String): RDD[MNISTLabels] = {

    val rawLabels = sc.binaryFiles(resolve(name))

    rawLabels.map(p => {
      val stream = p._2
      new MNISTLabels(stream.toArray())
    })
  }


  def asImageString(iNDArray: INDArray): String = {
    val arr = toArray(iNDArray)
    MNISTImage(arr.map(d => d.toByte), 28, 28).toString
  }


  protected def hdfs_save(rdd: RDD[MNISTImage], baseName: String): String = {
    val tmpName = baseName + "_" + UUID.randomUUID().toString
    rdd.saveAsObjectFile(tmpName)
    tmpName
  }
}
