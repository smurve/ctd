package org.smurve.mnist

import java.nio.ByteBuffer

import scala.collection.immutable.IndexedSeq


/**
  * A file containing MNIST Images
  * see: http://yann.lecun.com/exdb/mnist/
  */
class MNISTImages(bytes: Array[Byte] ) {

  val headerSize = 16
  val IMAGES: Int = 0x803 // magic number for image file
  val LABELS: Int = 0x801 // magic number for label file

  def asInt(bytes: Array[Byte]): Int = ByteBuffer.wrap(bytes).getInt

  val magicNumber: Int = asInt(bytes.slice(0,4))

  assert(magicNumber == IMAGES )

  val numImgs: Int = asInt(bytes.slice(4,8))
  val numRows: Int = asInt(bytes.slice(8,12))
  val numCols: Int = asInt(bytes.slice(12,16))

  val imgSize: Int = numRows * numCols

  def imgAtIndex (index: Int ) : MNISTImage =
    MNISTImage(bytes.slice( headerSize + imgSize * index, headerSize + imgSize * ( index + 1 )), numCols, numRows)

  lazy val imgs: IndexedSeq[MNISTImage] = ( 0 until numImgs).map(imgAtIndex)
}

