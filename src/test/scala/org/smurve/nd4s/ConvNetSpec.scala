package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FlatSpec, ShouldMatchers}
import org.nd4s.Implicits._

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

    val conv = Conv(theta1, depth_input = 2, height_input = 5, width_input = 5)
    val pool: Layer = (
      AvgPooling(depth_stride = 2, height_stride = 2, width_stride = 2)
        |:| Flatten(3, 2, 2)).asInstanceOf[AvgPooling]
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

      conv.dy_dx(0, 1, 2, 0, 1, 1) shouldEqual 0 // x_011 is not involved in computing y_012
      conv.dy_dx(0, 1, 2, 0, 1, 2) shouldEqual 1 // theta_010 = first LRF, second row (below the bias!), first column
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
      conv.dy_dTheta(5, 3, 2, 1, 2, 0, x) shouldEqual 0

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

  "An avg pool" should "produce the correctly averaged rank 3 tensor" in {
    new TestData {
      val y: INDArray = pool.fun(expected_conv_result)
      y shouldEqual expected_pool_result
    }
  }

  "An avg pool" should "calculate averages over width, height, and depth" in {
    new TestData {
      val poolNet: Layer = pool |:| output

      val input: INDArray = expected_conv_result
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
      val input: INDArray = expected_conv_result
      val y_bar: INDArray = expected_pool_result.ravel
      val propd: (INDArray, List[INDArray], Double) = poolNet.fwbw(input, y_bar)
      propd._1.sum(Array(0, 1, 2, 3): _*)(0) shouldEqual 0 // we're at the global minimum

      y_bar(4) = 2 // gives us dC/dy = (0,0,0,0,2,0,0,0,0,0,0,0) from the output layer
      val propd1: (INDArray, List[INDArray], Double) = poolNet.fwbw(x, y_bar)
      val dC_dx: INDArray = propd1._1
      dC_dx(0, 0, ->, ->) shouldEqual zeroMap
      dC_dx(0, 1, ->, ->) shouldEqual zeroMap
      dC_dx(1, 0, ->, ->) shouldEqual someDeriv
      dC_dx(1, 1, ->, ->) shouldEqual someDeriv
      dC_dx(2, 0, ->, ->) shouldEqual zeroMap
      dC_dx(2, 1, ->, ->) shouldEqual zeroMap
    }
  }

  "A conv net" should "build from basic layers and compute output vectors correctly" in {

    new TestData {
      val convnet: Layer = conv |:| pool |:| dense |:| output

      val y: INDArray = convnet.ffwd(x)

      y shouldEqual vec(32, -32)
    }
  }


  "A dense net" should "compute backprop dC/dx correctly" in {

    new TestData {

      val denseNet: Layer = Flatten(3, 2, 2) |:| dense |:| output

      val input: INDArray = expected_pool_result
      val y_bar: INDArray = vec(30, -32)

      val (from_backprop,_,_) = denseNet.fwbw(input, y_bar)
      val numerical: INDArray = df_dx(cost(denseNet, y_bar))(input)

      numerical shouldEqual from_backprop
    }
  }


  "A pool net" should "compute backprop dC/dx correctly" in {

    new TestData {

      val poolnet: Layer = pool |:| dense |:| output

      val input: INDArray = expected_conv_result
      val y_bar: INDArray = vec(30, -32)

      val (from_backprop,_,_) = poolnet.fwbw(input, y_bar)
      val numerical: INDArray = df_dx(cost(poolnet, y_bar))(input)

      numerical shouldEqual from_backprop
    }
  }


  "A conv net" should "compute backprop dC/dx correctly" in {

    new TestData {
      val convnet: Layer = conv |:| pool |:| dense |:| output

      val (id, ir, ic) = (1, 2, 2)
      val y_bar: INDArray = vec(30, -32)
      val epsilon = 1e-2
      val propd_m: PROPAGATED = convnet.fwbw(x, y_bar)
      x(id, ir, ic) = x(id, ir, ic) + epsilon
      val propd_r: PROPAGATED = convnet.fwbw(x, y_bar)
      val Cr: Double = propd_r._3

      x(id, ir, ic) = x(id, ir, ic) - epsilon
      val propd_l: PROPAGATED = convnet.fwbw(x, y_bar)
      val Cl: Double = propd_l._3

      val dC_dx_n: Double = (Cr - Cl) / 2 / epsilon // numeric
      val dC_dx_b: INDArray = propd_m._1 // backprop

      // compare backprop to numeric approximation
      dC_dx_b(id, ir, ic) shouldEqual dC_dx_n
    }
  }
}
