package org.smurve.mnist

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j


/**
  * A file containing MNIST Labels
  * see: http://yann.lecun.com/exdb/mnist/
  *
  */
class MNISTLabels( val bytes: Array[Byte] ) extends MNISTData with Serializable {

  //def bytes: Array[Byte] = data

  val headerSize = 8

  assert(magicNumber == LABELS )

  val numLabels: Int = asInt(bytes.slice(4,8))

  checkValues()

  def checkValues(): Unit = {
    for ( i <- 0 to 9) {
      require(bytes.count(_ == 0) > 100, s"Not enough ${i}")
    }
  }

  def lv ( index: Int ) : Int = bytes(headerSize+index)

  def labelAsArray(lval: Int): Array[Double] = (0 to 9).map( i=> if (lval == i) 1.0 else 0.0).toArray

  def lblForVal ( lval: Int ) : INDArray = Nd4j.create(labelAsArray(lval))

  def lblAtPos( index: Int ) : INDArray = {
    lblForVal(lv(index))
  }

  def arrayOfOneHotVectors: Array[Array[Double]] = {
    val onehots = Array.fill(numLabels){Array[Double]()}
    for (i <- onehots.indices) {
      onehots(i) = labelAsArray(bytes(headerSize + i))
    }
    onehots
  }

  lazy val asINDarray: INDArray = Nd4j.create(arrayOfOneHotVectors)

}

