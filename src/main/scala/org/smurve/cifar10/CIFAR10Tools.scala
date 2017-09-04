package org.smurve.cifar10

import java.io.DataInputStream

import com.sksamuel.scrimage.{Image, RGBColor}
import org.apache.spark.SparkContext
import org.apache.spark.input.PortableDataStream
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.smurve.cifar10.config.CIFAR10Config

import scala.language.postfixOps

trait CIFAR10Tools {

  protected val config: CIFAR10Config
  protected val session: SparkSession
  protected val sc: SparkContext

  val IMG_WIDTH = 32
  val IMG_HEIGHT = 32
  val NUM_CHANNELS = 3
  val CHANNEL_SIZE: Int = IMG_WIDTH * IMG_HEIGHT
  val BUFFER_SIZE_PER_ENTRY: Int = 1 + NUM_CHANNELS * CHANNEL_SIZE
  val NUM_RECORDS_PER_FILE = 10000

  val categories = Array(
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck")


  /**
    *
    * @param pattern a file pattern to be appended to the config's base path (prefix)
    * @return an RDD containing the file names and the associated portable data streams
    */
  def readData(pattern: String): RDD[(String, PortableDataStream)] = sc.binaryFiles(config.prefix + s"/$pattern")


  /**
    * read an entire data file into a pair on INDArrays
    * This file is assumed to contain 10000 records of size 3073
    * @param fileName the simple file name of the binary file
    * @return
    */
  def read (fileName: String): (INDArray, INDArray) = {
    val rdd = readData(fileName)
    val arr = rdd.collect.head._2.toArray.map(b => (b & 0xFF).toDouble)

    val ndarr = Nd4j.create(arr).reshape( NUM_RECORDS_PER_FILE, BUFFER_SIZE_PER_ENTRY)

    val samples = ndarr(->, 1->) / 255

    val labels = Nd4j.zeros(10 * NUM_RECORDS_PER_FILE).reshape(NUM_RECORDS_PER_FILE, 10)
    for ( i <- 0 until NUM_RECORDS_PER_FILE ) labels(i, ndarr(i, 0).toInt) = 1.0

    (samples, labels)
  }

  /**
    * read the next image out of an open stream: The structure is assumed to be 1 + 3 x 32 x 32.
    * 1 byte for the label and 3 x 1024 bytes for the three RGB Layers of the image
    *
    * @param stream an open data input stream
    * @return the image and a label
    */
  def nextImage(stream: DataInputStream): (Image, Int) = {
    val buffer = new Array[Byte](BUFFER_SIZE_PER_ENTRY)
    val check = stream.read(buffer)
    assert(check == BUFFER_SIZE_PER_ENTRY, s"Failed to read $BUFFER_SIZE_PER_ENTRY bytes. Got $check instead")
    val pixels = for (pos <- 1 to CHANNEL_SIZE) yield {
      val red = buffer(pos)
      val green = buffer(pos + CHANNEL_SIZE)
      val blue = buffer(pos + 2 * CHANNEL_SIZE)
      RGBColor(red & 0xFF, green & 0xFF, blue & 0xFF).toPixel
    }
    (Image(IMG_WIDTH, IMG_HEIGHT, pixels.toArray), buffer(0).toInt)
  }


}
