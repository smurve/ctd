package org.smurve.shapes

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.smurve.nd4s.vec

/**
  * @param n_symbols number of different 4x4 symbols to use, supporting up to 6 symbols
  * @param noise the noise of the background. Convergence will get pretty hard above 0.1 (10 percent).
  */
class ShapeData(n_symbols: Int = 6, noise: Double = 0.08 ) {

  val SYMBOL_SIZE = 4
  val cross: INDArray = vec(1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1).reshape(SYMBOL_SIZE, SYMBOL_SIZE)
  val circle: INDArray = vec(0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0).reshape(SYMBOL_SIZE, SYMBOL_SIZE)
  val plus: INDArray = vec(0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0).reshape(SYMBOL_SIZE, SYMBOL_SIZE)
  val triangle: INDArray = vec(1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0).reshape(SYMBOL_SIZE, SYMBOL_SIZE)
  val horiz: INDArray = vec(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0).reshape(SYMBOL_SIZE, SYMBOL_SIZE)
  val vert: INDArray = vec(0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0).reshape(SYMBOL_SIZE, SYMBOL_SIZE)
  val symbolMap = Array(cross, circle, plus, triangle, horiz, vert)

  def labelMap(i: Integer): INDArray = {
    require(i < n_symbols, s"Only supporting up to $n_symbols indices")
    vec((0 until n_symbols).map(_.toDouble).toArray: _*) === i
  }

  def createBatch(imgSize: Int, batchSize: Int, depth: Int): (INDArray, INDArray) = {

    val maxPos = imgSize - SYMBOL_SIZE
    val (samples, labels) = (
      Nd4j.zeros(batchSize, depth, imgSize, imgSize),
      Nd4j.zeros(batchSize, n_symbols)
    )
    for (i <- 0 until batchSize) {
      val symbol = (math.random * n_symbols).toInt
      val posX = (math.random * (maxPos + 1)).toInt
      val posY = (math.random * (maxPos + 1)).toInt
      samples(i) = createImage(imgSize, imgSize, depth, symbolMap(symbol))(posX, posY)
      labels(i) = labelMap(symbol)
    }

    (samples, labels)
  }

  def createImage(width: Int, height: Int, depth: Int, symbol: INDArray)(posx: Int, posy: Int): INDArray = {
    require(width > 0 && height > 0 && posx >= 0 && posy >= 0, "dimensions and position must be positive integers")
    require(height >= SYMBOL_SIZE && width >= SYMBOL_SIZE, "dimensions must allow a shape to fit in.")
    require(SYMBOL_SIZE + posx <= width && SYMBOL_SIZE + posy <= height, "Can't put the entire shape at this position")

    val img = Nd4j.zeros(1, width, height)
    for {
      x <- 0 until SYMBOL_SIZE
      y <- 0 until SYMBOL_SIZE
    } {
      img(0, posx + x, posy + y) = symbol(x, y)
    }
    val bg1 = Nd4j.rand(Array(1, width, height)) * noise
    val bg2 = Nd4j.rand(Array(1, width, height)) * noise

    val layers = List(img, bg1, bg2).map(l => (l, math.random)).sortBy(_._2).map(_._1).toArray

    Nd4j.hstack(layers: _*)

  }
}
