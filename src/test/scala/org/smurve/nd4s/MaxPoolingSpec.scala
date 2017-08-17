package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.scalatest.{FlatSpec, ShouldMatchers}

class MaxPoolingSpec extends FlatSpec with ShouldMatchers with TestTools{

  trait TestData {
    val input: INDArray = vec(
      5, 1, 6, 4,
      2, 3, 112, 5,
      5, 4, 7, 6,
      4, 5, 8, 7,

      111, 2, 3, 4,
      2, 3, 4, 5,
      3, 9, 5, 6,
      4, 121, 0, 122,

      1, 2, 9, 4,
      2, 211, 7, 5,
      3, 6, 5, 222,
      221, 0, 6, 9,

      1, 2, 9, 4,
      2, 3, 212, 1,
      3, 7, 2, 6,
      4, 5, 6, 7,

      1, 2, 5, 4,
      3, 9, 312, 5,
      2, 4, 322, 6,
      4, 5, 7, 1,

      4, 5, 6, 7,
      3, 311, 8, 6,
      2, 2, 9, 5,
      8, 321, 3, 1
    ).reshape(3, 2, 4, 4)

    val expected_pool_result: INDArray = vec(
      111, 112,
      121, 122,

      211, 212,
      221, 222,

      311, 312,
      321, 322
    ).reshape(3, 2, 2)

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

    val pool: Layer = (
      MaxPool(depth_stride = 2, height_stride = 2, width_stride = 2)
        |:| Flatten(3, 2, 2)).asInstanceOf[MaxPool]
    val dense = FCL(theta2)
    val output = Euclidean()
  }



  "A max pool" should "produce the correctly computed rank 3 tensor" in {
    new TestData {
      val y: INDArray = pool.fun(input)
      y shouldEqual expected_pool_result
    }
  }

  "A max pool" should "feed forward by applying the max function" in {
    new TestData {
      val poolNet: Layer = pool |:| output

      poolNet.ffwd(input) shouldEqual expected_pool_result
    }
  }


  "A max pooling layer" should "compute backprop dC/dx correctly" in {

    new TestData {
      val poolnet: Layer = pool |:| dense |:| output
      val y_bar: INDArray = vec(2598, -2596)

      validateBackProp(poolnet, input, y_bar)
    }
  }


}
