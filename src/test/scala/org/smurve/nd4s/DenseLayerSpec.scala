package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.scalatest.{FlatSpec, ShouldMatchers}

class DenseLayerSpec extends FlatSpec with ShouldMatchers with TestTools{

  trait TestData {

    val input: INDArray = vec(0, 0, 0, 0, 4, 4, 4, 4, 3, 5, 3, 5).reshape(1,12)

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

  "It" should "exhibit a number of symmetries" in {
    new TestData {
      val input2: INDArray = vec(
        0, 0, 0, 0, 4, 4, 4, 4, 3, 5, 3, 5,
        1, 1, 1, 1, 2, 2, 3, 3, 2, 1, 1, 2
      ).reshape(2, 12)
      val y_bar2: INDArray = vec(1,2,3,4).reshape(2,2)
      val denseNet: Layer = dense |:| output
      checkSymmetries(denseNet, input2, y_bar2)
    }
  }


}
