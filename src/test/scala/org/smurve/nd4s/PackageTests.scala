package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.scalatest.{FlatSpec, ShouldMatchers}

import scala.util.Random

class PackageTests extends FlatSpec with ShouldMatchers {

  val seed = 123

  "shuffle" should "shuffle - obviously." in {

    val orig1 = (1 to 12).asNDArray(4,3)
    val orig2 = (13 to 24).asNDArray(4,3)
    val shuffled = shuffle((orig1, orig2), new Random(seed))
    shuffled._1(0,0) should equal(7)
    shuffled._2(0,0) should equal(19)
  }

  val input: INDArray = vec(
    1,2,3,
    4,5,6,
    7,8,9,

    1,2,0,
    4,10,6,
    7,8,9).reshape(2,3,3)

  "extract()" should "extract max/min values correctly " in {

    reduceByElimination(input, iArr(input), _>_)._1 shouldEqual 10.0
    reduceByElimination(input, iArr(input), _>_)._2 shouldEqual Array(1,1,1)

    reduceByElimination(input, iArr(input), _<_)._1 shouldEqual 0.0
    reduceByElimination(input, iArr(input), _<_)._2 shouldEqual Array(1,0,2)

    reduceByElimination(input.ravel, iArr(input.ravel), _>_)._1 shouldEqual 10.0
    reduceByElimination(input.ravel, iArr(input.ravel), _>_)._2 shouldEqual Array(0, 13)

  }

}
