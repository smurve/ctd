package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FlatSpec, ShouldMatchers}
import org.nd4s.Implicits._

class FCLSpec extends FlatSpec with ShouldMatchers {

  val theta: INDArray = Nd4j.create(Array(1, 1, 1, -2f, -1, 0, 2, 3, 4)).reshape(3, 3)

  "h1" should "pad horizontically" in {
    val theta = Nd4j.create(Array(2f, 5, 8, 2, 9, 16)).reshape(2, 3)
    h1(theta) shouldEqual Nd4j.create(Array(1, 2f, 5, 8, 1, 2, 9, 16)).reshape(2, 4)
  }

  "v1" should "pad vertically" in {
    val theta = Nd4j.create(Array(-2f, -1, 0, 2, 3, 4)).reshape(2, 3)
    v1(theta) shouldEqual Nd4j.create(Array(1, 1, 1, -2f, -1, 0, 2, 3, 4)).reshape(3, 3)
  }


  "An FCL feed forward" should "simmply perform matrix multiplication on the batch" in {
    val x = Nd4j.create(Array(1f, 2, 3, 4)).reshape(2, 2)
    val network = FCL(theta) |:| Euclidean()
    val y = network.ffwd(x)
    y shouldEqual Nd4j.create(Array(3f, 6, 9, 3, 10, 17)).reshape(2, 3)
  }

  lazy val propd: PROPAGATED = {
    val x = Nd4j.create(Array(1f, 2, 3, 4)).reshape(2, 2)
    val output = Euclidean()

    val y_bar = Nd4j.create(Array(2f, 6, 5, 3, 4, 6)).reshape(2, 3)

    val network = FCL(theta) |:| output
    network.fwbw(x, y_bar)
  }

  "A Euclidean" should "compute the cost correctly" in {
    val y = Nd4j.create(Array(2f, 5, 8, 2, 9, 16)).reshape(2,3)
    val y_bar = Nd4j.create(Array(2f, 6, 5, 3, 4, 6)).reshape(2, 3)
    val output = Euclidean()
    output.c(y, y_bar) shouldEqual 68
    output.grad_c(y, y_bar) shouldEqual Nd4j.create(Array(0f, -1, 3, -1, 5, 10)).reshape(2,3)
  }

  "An FCL forward-backward pass" should "return the gradients and the current cost of the batch" in {

    propd._1 shouldEqual Nd4j.create(Array(-2f, 18, -6, 62)).reshape(2,2)
    propd._2 shouldEqual List(Nd4j.create(Array(1f, 6, 15, 1, 18, 37, 2, 24, 52)).reshape(3,3))
    propd._3 shouldEqual 87
  }

  "An FCL" should "calculate correct gradients with regards to x" in {
    val epsilon = 1e-4
    val x = Nd4j.create(Array(1f, 2, 3, 4)).reshape(2, 2)
    val dx1 = Nd4j.create(Array(0, epsilon, 0, 0)).reshape(2, 2)
    val y_bar = Nd4j.create(Array(2f, 6, 5, 3, 4, 6)).reshape(2, 3)

    val output = Euclidean()

    val f = FCL(theta) |:| output

    val grad = (f.fwbw(x + dx1, y_bar)._3 - f.fwbw(x - dx1, y_bar)._3) / 2 / epsilon
    val grad1 = f.fwbw(x, y_bar)._1
    math.abs(grad - grad1(0,1))/grad should be < math.sqrt(epsilon)
  }

  "An FCL" should "calculate correct gradients with regards to theta" in {
    val epsilon = 1e-4
    val x = Nd4j.create(Array(1f, 2, 3, 4)).reshape(2, 2)
    val output = Euclidean()

    val y_bar = Nd4j.create(Array(2f, 6, 5, 3, 4, 6)).reshape(2, 3)

    val fcl = FCL(theta)
    val f = fcl |:| output
    fcl += Nd4j.create(Array(0,0,0, 0, epsilon, 0, 0, 0, 0)).reshape(3,3)
    val c_right = f.fwbw(x, y_bar)._3
    fcl += Nd4j.create(Array(0,0,0, 0, -2 * epsilon, 0, 0, 0, 0)).reshape(3,3)
    val c_left = f.fwbw(x, y_bar)._3

    val grad = (c_right - c_left) / 2 / epsilon
    val grad1 = f.fwbw(x, y_bar)._2.head
    math.abs(grad - grad1(1,1))/grad should be < math.sqrt(epsilon)
  }
}
