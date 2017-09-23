package org.smurve.cifar10

import java.io.{File, FileInputStream}

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import DataContext._
import org.nd4j.linalg.api.buffer.DataBuffer

import scala.language.postfixOps
import org.smurve.cifar10._

/**
  * creates a context for local execution with files read from local file system
  */
class DataContext(basePath: String ) {


  def read(fileName: String, num_records: Int = NUM_RECORDS_PER_FILE, dump: Boolean = false): (INDArray, INDArray) = {

    val bytes = new Array[Byte](num_records * BUFFER_SIZE_PER_ENTRY)
    val fis = new FileInputStream(new File(basePath + "/" + fileName))
    fis.read(bytes)

    val arr = bytes.map(b => (b & 0xFF).toFloat / 256 - 0.5)

    val ndarr = Nd4j.create(arr)
    val asRecords = ndarr.reshape(num_records, BUFFER_SIZE_PER_ENTRY)
    val reduced = asRecords(0 -> num_records, ->)

    /*

    This produces the amazing off-by-one bug

    val samples = reduced(->, 1 ->).reshape(num_records, NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
    val centered = samples / 255 - 0.5
    */
    val samples = reduced(->, 1 ->).reshape(num_records, NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

    val labels = Nd4j.zeros(10 * num_records).reshape(num_records, 10)
    for (i <- 0 until num_records)  {
      val value = bytes(i * BUFFER_SIZE_PER_ENTRY).toInt
      labels(i, value) = 1.0
    }

    if ( dump ) {
      for ( i <- 0 to 5 ) {
        val orig = (1 to 5).map(n=>bytes(BUFFER_SIZE_PER_ENTRY * i + n) & 0xFF).toList.toString
        val samp = (0 to 4).map(n=>((samples(i, 0, 0, n) + 0.5) * 256).toInt).toList.toString
        println(s"$orig == $samp")
      }
      dumpAsImages( samples, labels, 100 )
    }

    (samples, labels)
  }


  def readSplit(fileName: String, num_records: Int = NUM_RECORDS_PER_FILE, dump: Boolean = false): (INDArray, INDArray) = {

    val img_bytes = new Array[Byte](num_records * IMG_SIZE)
    val img_fis = new FileInputStream(new File(basePath + "/img_" + fileName))
    img_fis.read(img_bytes)

    val arr = img_bytes.map(b => (b & 0xFF).toFloat) //  / 256f - 0.5
    val images = Nd4j.create(arr).reshape(num_records, NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
    images.divi(256f).subi(.5f)

    val lbl_bytes = new Array[Byte](num_records)
    val lbl_fis = new FileInputStream(new File(basePath + "/lbl_" + fileName))
    lbl_fis.read(lbl_bytes)
    val labels = Nd4j.zeros (num_records * 10).reshape( num_records, 10)
    for (i <- 0 until num_records)  {
      val value = lbl_bytes(i).toInt
      labels(i, value) = 1.0
    }


    if ( dump ) {
      dumpAsImages(images, labels, 100)
    }

    (images, labels)
  }

}



object DataContext {
  val IMG_WIDTH = 32
  val IMG_HEIGHT = 32
  val NUM_CHANNELS = 3
  val IMG_SIZE: Int = NUM_CHANNELS * IMG_WIDTH * IMG_HEIGHT
  val CHANNEL_SIZE: Int = IMG_WIDTH * IMG_HEIGHT
  val BUFFER_SIZE_PER_ENTRY: Int = 1 + NUM_CHANNELS * CHANNEL_SIZE
  val NUM_RECORDS_PER_FILE = 10000
}
