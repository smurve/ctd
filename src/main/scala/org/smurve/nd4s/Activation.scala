package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._

/**
  * convenience class that delegates vector f_prime to the Double f_prime
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
  def f_prime (x: INDArray): INDArray = appf(x, f_prime)

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

  /**
    * nothing to do here, pass on
    * @param grads the gradients to be applied (but not here).
    */
  override def update(grads: Seq[INDArray]): Unit = nextLayer.update(grads)
}
