package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.scalatest.{FlatSpec, ShouldMatchers}

class MaxPoolingSpec extends FlatSpec with ShouldMatchers with TestTools{

  trait TestData {
    val input: INDArray = vec(
      -2, -2, -2, -2,
      -2, -2, -2, -2,
      -2, -2, -2, -2,
      -2, -2, -2, -2,

      2, 2, 2, 2,
      2, 2, 2, 2,
      2, 2, 2, 2,
      2, 2, 2, 2,

      4, 4, 4, 4,
      4, 4, 4, 4,
      4, 4, 4, 4,
      4, 4, 4, 4,

      4, 4, 4, 4,
      4, 4, 4, 4,
      4, 4, 4, 4,
      4, 4, 4, 4,

      1, 2, 3, 4,
      2, 3, 4, 5,
      3, 4, 5, 6,
      4, 5, 6, 7,

      4, 5, 6, 7,
      3, 4, 5, 6,
      2, 3, 4, 5,
      1, 2, 3, 4
    ).reshape(3, 2, 4, 4)

    val expected_pool_result: INDArray = vec(
      2, 2,
      2, 2,

      4, 4,
      4, 4,

      5, 7,
      5, 7
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
      AvgPool(depth_stride = 2, height_stride = 2, width_stride = 2)
        |:| Flatten(3, 2, 2)).asInstanceOf[AvgPool]
    val dense = FCL(theta2)
    val output = Euclidean()
  }


  "An avg pool" should "produce the correctly averaged rank 3 tensor" in {
    new TestData {
      val y: INDArray = pool.fun(input)
      y shouldEqual expected_pool_result
    }
  }

  "An avg pool" should "calculate averages over width, height, and depth" in {
    new TestData {
      val poolNet: Layer = pool |:| output

      poolNet.ffwd(input) shouldEqual expected_pool_result.ravel
    }
  }

  "An avg pool" should "calculate the correct partial derivatives" in {
    new TestData {
      val poolNet: Layer = pool |:| output
      val zeroMap: INDArray = Nd4j.zeros(4, 4)
      val someDeriv: INDArray = vec(
        .25, .25, 0, 0,
        .25, .25, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0).reshape(4, 4)
      val y_bar: INDArray = expected_pool_result.ravel
      val propd: (INDArray, List[INDArray], Double) = poolNet.fwbw(input, y_bar)
      propd._1.sum(Array(0, 1, 2, 3): _*)(0) shouldEqual 0.0 // we're at the global minimum

      y_bar(4) = 2 // gives us dC/dy = (0,0,0,0,2,0,0,0,0,0,0,0) from the output layer
      val propd1: (INDArray, List[INDArray], Double) = poolNet.fwbw(input, y_bar)
      val dC_dx: INDArray = propd1._1
      dC_dx(0, 0, ->, ->) shouldEqual zeroMap
      dC_dx(0, 1, ->, ->) shouldEqual zeroMap
      dC_dx(1, 0, ->, ->) shouldEqual someDeriv
      dC_dx(1, 1, ->, ->) shouldEqual someDeriv
      dC_dx(2, 0, ->, ->) shouldEqual zeroMap
      dC_dx(2, 1, ->, ->) shouldEqual zeroMap
    }
  }

  "An average pooling layer" should "compute backprop dC/dx correctly" in {

    new TestData {
      val poolnet: Layer = pool |:| dense |:| output
      val y_bar: INDArray = vec(30, -32)

      validateBackProp(poolnet, input, y_bar)
    }
  }


}
