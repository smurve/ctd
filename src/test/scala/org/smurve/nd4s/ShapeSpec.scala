package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FlatSpec, ShouldMatchers}
import org.nd4s.Implicits._

class ShapeSpec extends FlatSpec with ShouldMatchers with TestTools {

  val input: INDArray = vec(1,2,3,4,5,6,7,8, 2,3,4,5,6,7,8,9, 3,4,5,6,7,8,9,0)

  "A Shape layer" should "reshape any number of input vectors correctly" in {

    val network = Shape(2,2,2) |:| Euclidean()

    val y = network.ffwd(input)
    y shouldEqual input.reshape(3,2,2,2)

    val (dC_dy, grads, c) = network.fwbw(input, input - 1)
    dC_dy shouldEqual Nd4j.ones(3,2,2,2)
    grads should be ('empty)
  }
}
