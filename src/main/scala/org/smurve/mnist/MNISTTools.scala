package org.smurve.mnist

import java.io.{File, FileInputStream}

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.smurve.nd4s._

import scala.util.Random

trait MNISTTools  {

  val NUM_BYTES_PER_IMAGE_FILE = 47040016
  val NUM_BYTES_PER_LABEL_FILE = 60008

  def saveModel(name: String, weights: Map[String, INDArray] ) : Unit = {

    println(s"Saving model as $name")
    for ( theta <- weights ) {
      Nd4j.saveBinary(theta._2, new File ( s"${name}_${theta._1}"))
    }
  }

  def readModel(name: String, weightNames: List[String]): Map[String, INDArray] = {
    val res = for (wn <- weightNames) yield {
      val weights = Nd4j.readBinary( new File(s"${name}_$wn"))
      wn->weights
    }
    res.toMap
  }

  /**
    * @param name name of file or directory to read from
    * @return an RDD of MNISTImages wrappers
    */
  protected def createImagesFromBinary(name: String): MNISTImages = {

      val stream = new FileInputStream(new File(name))
      val bytes = new Array[Byte](NUM_BYTES_PER_IMAGE_FILE)
      stream.read(bytes)
      new MNISTImages(bytes)
  }



  /**
    * @param name name of file or directory to read from
    * @return an RDD of MNISTLabel wrappers
    */
  protected def createLabelsFromBinary(name: String): MNISTLabels = {

    val stream = new FileInputStream(new File(name))
    val bytes = new Array[Byte](NUM_BYTES_PER_LABEL_FILE)
    stream.read(bytes)
    new MNISTLabels(bytes)

  }

  /**
    * Read the data files
    * @param key either "train" or "test"
    * @return a pair of INDArrays containing images and labels
    */
  def readFromBinary(key: String): (INDArray, INDArray) = {
    val images: MNISTImages = createImagesFromBinary(s"input/$key")
    val labels = createLabelsFromBinary(s"input/$key-labels")

    (images.asINDarray, labels.asINDarray)
  }


  def asImageString(iNDArray: INDArray): String = {
    val arr = toArray(iNDArray)
    MNISTImage(arr.map(d => d.toByte), 28, 28).toString
  }

  /**
    *
    * @param numTrain number of training records/labels to read from file
    * @param numTest number of test records/labels to read from file
    * @param rnd a random generator
    * @return a pair of pairs containing training samples and labels and test samples and labels
    */
  def  readMNIST(numTrain: Int, numTest: Int, rnd: Random = new Random()
                ): ((INDArray, INDArray), Option[(INDArray, INDArray)]) = {

    println("Reading images from file...")
    val (img_test, lbl_test) = readFromBinary("test")
    val trainingSet_orig = readFromBinary("train")
    println("Done.")
    println("Shuffling...")
    val (img_train, lbl_train) = shuffle(trainingSet_orig, random = rnd)
    println("Done.")
    val trainingSet = (img_train(0 -> numTrain, ->), lbl_train(0 -> numTrain, ->))
    val testSet = if (numTest > 0 )
      Some((img_test(0 -> numTest, ->), lbl_test(0 -> numTest, ->)))
    else
      None

    (trainingSet, testSet)
  }

}
