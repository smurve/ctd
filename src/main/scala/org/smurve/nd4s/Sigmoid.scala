package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

/**
  * sigmoid activation is an Activation but doesn't need the help of the Activation
  */
case class Sigmoid() extends Layer {
  /**
    * the function associated with this layer
    *
    * @param x the input vector
    * @return the function applied to the input vector
    */
  override def fun(x: INDArray): INDArray = sigmoid(x)

  def f_prime (x: INDArray): INDArray = {
    val s = sigmoid(x)
    val one = Nd4j.ones(x.size(0), x.size(1))
    s * ( one - s)
  }

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
    * nothing to do. Just pass on
    * @param grads gradients for the parameter layers to update
    */
  override def update(grads: List[INDArray]): Unit = nextLayer.update(grads)
}
