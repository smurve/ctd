package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.scalatest.{FlatSpec, ShouldMatchers}

class ConvNetSpec extends FlatSpec with ShouldMatchers with TestTools{

  trait TestData {
    val x: INDArray = vec(
      0, 1, 2, 3, 4,
      1, 2, 3, 4, 5,
      2, 3, 4, 5, 6,
      3, 4, 5, 6, 7,
      4, 5, 6, 7, 8,

      4, 5, 6, 7, 8,
      3, 4, 5, 6, 7,
      2, 3, 4, 5, 6,
      1, 2, 3, 4, 5,
      0, 1, 2, 3, 4
    ).reshape(2, 5, 5)

    val theta1: INDArray = vec(
      0, 0,
      1, 1,
      -1, -1,

      0, 0,
      -2, 2,
      -2, 2,

      0, 0,
      1, -.5,
      -.5, 1
    ).reshape(3, 3, 2)

    val expected_conv_result: INDArray = vec(
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

    // happens to have the same shape, of course
    val fake_dC_dy: INDArray = expected_conv_result

    val expected_pool_result: INDArray = vec(
      0, 0,
      0, 0,

      4, 4,
      4, 4,

      3, 5,
      3, 5
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

    val conv = Conv(theta1, depth_input = 2, height_input = 5, width_input = 5, height_theta = 2)
    val pool: Layer = (
      AvgPool(depth_stride = 2, height_stride = 2, width_stride = 2)
        |:| Flatten(3, 2, 2)).asInstanceOf[AvgPool]
    val dense = FCL(theta2)
    val output = Euclidean()
  }


  "A euclidean output layer" should "compute the L2 norm of the difference" in {
    new TestData {
      val propd: PROPAGATED = output.fwbw(vec(1, 2), vec(0, 0))
      propd._3 shouldEqual 2.5
      val dC_dx: INDArray = propd._1
      dC_dx shouldEqual vec(1, 2)
    }
  }


  "A conv layer" should "produce 6 feature matrices for 2 channels and 3 LRFs" in {
    new TestData {
      val y: INDArray = conv.fun(x)
      y shouldEqual expected_conv_result

      conv.dy_dx(0, 1, 2, 0, 1, 1) shouldEqual 0.0 // x_011 is not involved in computing y_012
      conv.dy_dx(0, 1, 2, 0, 1, 2) shouldEqual 1.0 // theta_010 = first LRF, second row (below the bias!), first column
    }
  }

  "A conv layer" should "compute dy/dx as the weights involved to produce y from x" in {
    new TestData {

      conv.dy_dx(0, 1, 2, 0, 1, 2) shouldEqual theta1(0, 1, 0)
      conv.dy_dx(0, 1, 2, 0, 1, 3) shouldEqual theta1(0, 1, 1)
      conv.dy_dx(0, 1, 2, 0, 2, 2) shouldEqual theta1(0, 2, 0)
      conv.dy_dx(0, 1, 2, 0, 2, 3) shouldEqual theta1(0, 2, 1)

      conv.dy_dx(5, 3, 3, 1, 3, 3) shouldEqual theta1(2, 1, 0)
      conv.dy_dx(5, 3, 3, 1, 3, 4) shouldEqual theta1(2, 1, 1)
      conv.dy_dx(5, 3, 3, 1, 4, 3) shouldEqual theta1(2, 2, 0)
      conv.dy_dx(5, 3, 3, 1, 4, 4) shouldEqual theta1(2, 2, 1)
    }
  }


  "A conv layer" should "compute dy/dtheta as the x values involved to produce y from theta" in {

    new TestData {
      conv.dy_dTheta(5, 3, 2, 1, 2, 0, x) shouldEqual 0.0

      conv.dy_dTheta(5, 3, 2, 2, 1, 0, x) shouldEqual x(1, 3, 2)
      conv.dy_dTheta(5, 3, 2, 2, 1, 1, x) shouldEqual x(1, 3, 3)
      conv.dy_dTheta(5, 3, 2, 2, 2, 0, x) shouldEqual x(1, 4, 2)
      conv.dy_dTheta(5, 3, 2, 2, 2, 1, x) shouldEqual x(1, 4, 3)
    }
  }

  "A conv layer" should "compute dC/dx correctly" in {

    new TestData {
      val dC_dx: INDArray = conv.dC_dx(x, fake_dC_dy)
    }
  }

  "A conv net" should "build from basic layers and compute output vectors correctly" in {

    new TestData {
      val convnet: Layer = conv |:| pool |:| dense |:| output

      val y: INDArray = convnet.ffwd(x)

      y shouldEqual vec(32, -32)
    }
  }


  "A conv net" should "compute backprop dC/dx correctly" in {
    new TestData {
      val convnet: Layer = conv |:| pool |:| dense |:| output
      val y_bar: INDArray = vec(30, -32)

      validateBackProp(convnet, x, y_bar)
    }
  }

}
