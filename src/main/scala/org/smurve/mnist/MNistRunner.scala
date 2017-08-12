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


abstract class MNistRunner(protected val config: MNistConfig) {

  protected val session: SparkSession
  protected val sc: SparkContext

  trait Params {

    val N_TRAINING = 6000
    val N_TEST = 1000
    val N_EPOCHS = 5
    val reportEvery = 10
    val eta: Double = 1e-4
    val nbatches = 100
    val parallelism = 4
    val task = false // task or data parallelism

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

    }

    session.stop()
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
