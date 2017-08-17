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

}
