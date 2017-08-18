package org.smurve.nd4s

import org.scalatest.{FlatSpec, ShouldMatchers}

class TestToolsSpec extends FlatSpec with ShouldMatchers with TestTools{

  override val PRECISION = Precision(1e-3)

  "double comparison" should "be up to a given precision" in {
    100.0 shouldEqual 100.0002

    // ...but
    doubleEq.areEqual(100.0, 101.0) should be ( false)
  }

  "INDArrays" should "compare by their double content" in {
    vec(1,2,3,4.001) shouldEqual vec(1,2,3,4)
    // ...but
    nd4jEq.areEqual(vec(1,2,3,4), vec(1,2,3,4.01)) should be ( false)
  }

  "sliceR" should "slice maintining the rank" in {
    val input = vec (1,2,3,4,1,2,3,4, 5,6,7,8,5,6,7,8, 1,3,5,7,1,3,5,7)

    val input0 = input.reshape(3,8)
    sliceR(input0, 0).rank shouldEqual 2
    sliceR(input0, 0) shouldEqual vec(1,2,3,4,1,2,3,4).reshape(1,8)
    sliceR(input0, 1) shouldEqual vec(5,6,7,8,5,6,7,8).reshape(1,8)

    val input1 = input.reshape(3,2,4)
    sliceR(input1, 0).rank shouldEqual 3
    sliceR(input1, 0) shouldEqual vec(1,2,3,4,1,2,3,4).reshape(1,2,4)
    sliceR(input1, 1) shouldEqual vec(5,6,7,8,5,6,7,8).reshape(1,2,4)

    val input2 = input.reshape(3,2,2,2)
    sliceR(input2, 0).rank shouldEqual 4
    sliceR(input2, 0) shouldEqual vec(1,2,3,4,1,2,3,4).reshape(1,2,2,2)
    sliceR(input2, 1) shouldEqual vec(5,6,7,8,5,6,7,8).reshape(1,2,2,2)

  }
}
