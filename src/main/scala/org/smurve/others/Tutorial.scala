package org.smurve.others
import org.nd4s.Implicits._
import org.nd4j.linalg.factory.Nd4j

object Tutorial {


  def main(args: Array[String]): Unit = {
    playAround()
  }

  def playAround(): Unit = {
    val ones = Nd4j.ones(16)
    val ones4x4 = ones.reshape(4,4)
    println(ones4x4)
  }

}
