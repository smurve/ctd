package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._

/**
  */
trait Activation extends Layer {

  /**
    * the derivative function to be implemented by you
    * @param x input value
    * @return f_prime applied to x
    */
  def f_prime (x: Double): Double

  /**
    * the element-wise function application for the derivative
    * @param x the input vector
    * @return f_prime applied to every element of x
    */
  def f_prime (x: INDArray): INDArray = appf(x, d=>f_prime(d))

  /**
    * forward pass and back propagation in one method call
    *
    * @param x     the batch of input row vectors
    * @param y_bar the batch of expected outcome row vectors
    */
  override def fwbw(x: INDArray, y_bar: INDArray): (INDArray, List[INDArray], Double) = {
    val (delta, grads, c) = nextLayer.fwbw(fun(x), y_bar)
    (delta * f_prime(x), grads, c)
  }
}
