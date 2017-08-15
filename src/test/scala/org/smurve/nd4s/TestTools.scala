package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.scalactic.Equality
import org.nd4s.Implicits._

trait TestTools {

  case class Precision(epsilon: Double)

  // to avoid potential interference by making a Double implicit
  implicit val PRECISION: Precision = Precision(1e-3)

  implicit val doubleEq: Equality[Double] = new Equality[Double] {
    override def areEqual(a: Double, b: Any): Boolean = a - b.asInstanceOf[Double] < PRECISION.epsilon
  }

  /**
    * INDArray equality to a certain precision
    */
  implicit val nd4jEq: Equality[INDArray] = new Equality[INDArray] {
    override def areEqual(a: INDArray, b: Any): Boolean = a - b.asInstanceOf[INDArray] < PRECISION.epsilon
  }

  /**
    * the cost function
    * @param network the network to use
    * @param y_bar the expected output
    * @return the cost at the given input
    */
  def cost(network: Layer, y_bar: INDArray): INDArray => Double = network.fwbw(_, y_bar)._3


  /**
    * numerically compute the gradient with regards to x
    *
    * @param f the function
    * @param x the input
    * @return
    */
  def df_dx(f: (INDArray) => Double)(x: INDArray)(implicit precision: Precision): INDArray = {
    val l = x.length
    def epsvec(k: Int) = {
      val res = Nd4j.zeros(l)
      res(k) = precision.epsilon
      res
    }

    val res = Nd4j.zeros(l)
    for (i <- 0 until x.length()) {
      res(i) = (f(x + epsvec(i)) - f(x - epsvec(i))) / 2 / precision.epsilon
    }
    res.reshape(x.shape: _*)
  }


}
