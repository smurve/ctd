package org.smurve.mnist

import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}

/**
  * Represents a file containing MNIST data
  * see: http://yann.lecun.com/exdb/mnist/
  */
abstract class MNISTFile ( fileName: String ){

  val bytes: Array[Byte] = Files.readAllBytes(Paths.get(fileName))

  val IMAGES: Int = 0x803 // magic number for image file
  val LABELS: Int = 0x801 // magic number for label file

  def asInt(bytes: Array[Byte]): Int = ByteBuffer.wrap(bytes).getInt

  val magicNumber: Int = asInt(bytes.slice(0,4))
  val headerSize: Int
}
