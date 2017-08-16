package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.scalatest.{FlatSpec, ShouldMatchers}

class DenseLayerSpec extends FlatSpec with ShouldMatchers with TestTools{

  trait TestData {

    val input: INDArray = vec(0, 0, 0, 0, 4, 4, 4, 4, 3, 5, 3, 5)

    val theta2: INDArray = vec(
      0, 0,
      1, -1,
      1, -1,
      1, -1,
      1, -1,
      1, -1,
      1, -1,
      1, -1,
      1, -1,
      1, -1,
      1, -1,
      1, -1,
      1, -1
    ).reshape(13, 2)

    val dense = FCL(theta2)
    val output = Euclidean()
  }


  "A dense layer" should "compute backprop dC/dx correctly" in {

    new TestData {
      val denseNet: Layer = dense |:| output
      val y_bar: INDArray = vec(30, -31)

      validateBackProp(denseNet, input, y_bar)
    }
  }


}
