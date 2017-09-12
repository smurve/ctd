package org.smurve.cifar10.runner

import java.io.{File, FileInputStream}

import org.deeplearning4j.datasets.iterator.callbacks.FileCallback
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

import scala.language.postfixOps

class CIFAR10LocalFileReader(numRecords: Int) extends FileCallback {

  val NUM_CHANNELS: Int = 3
  val IMG_HEIGHT = 32
  val IMG_WIDTH = 32
  val RECORD_SIZE: Int = NUM_CHANNELS * IMG_HEIGHT * IMG_WIDTH + 1

  override def call[T](file: File): T = {

    val arr = new Array[Byte](RECORD_SIZE * numRecords)

    val fis = new FileInputStream(file)

    if (fis.read(arr) != RECORD_SIZE * numRecords)
      throw new IllegalStateException("Couldn't read data from file")

    val ndarr =  Nd4j.create(arr.map(_.toDouble / 255)).reshape( numRecords, RECORD_SIZE)

    val samples = ndarr(->, 1->).reshape(numRecords, NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH) / 255

    val labels = Nd4j.zeros(10 * numRecords).reshape(numRecords, 10)
    for ( i <- 0 until numRecords ) labels(i, ndarr(i, 0).toInt) = 1.0

    new DataSet(samples, labels).asInstanceOf[T]
  }
}
