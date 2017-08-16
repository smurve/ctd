package org.smurve.nd4s

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
    println("hello")
  }
}
