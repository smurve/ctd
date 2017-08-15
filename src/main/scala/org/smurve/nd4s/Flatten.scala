package org.smurve.nd4s
import org.nd4j.linalg.api.ndarray.INDArray

/**
  * Do nothing but flatten the input vector in all forward passes
  */
case class Flatten(input_depth: Int, input_height: Int, input_width: Int ) extends Layer {

  /**
    * just flatten the layer
    *
    * @param x the input vector
    * @return the function applied to the input vector
    */
  override def fun(x: INDArray): INDArray = x.ravel

  /**
    * forward pass and back propagation in one method call
    *
    * @param x     the batch of input row vectors
    * @param y_bar the batch of expected outcome row vectors
    */
  override def fwbw(x: INDArray, y_bar: INDArray): (INDArray, List[INDArray], Double) = {
    val (dC_dx, grads, c) = nextLayer.fwbw(fun(x), y_bar)
    (dC_dx.reshape(input_depth, input_height, input_width), grads, c)
  }

  /**
    * update the weights using the head of the Seq
    * implementers must forward the tail to the subsequent layers
    *
    * @param grads the amount to be added
    */
  override def update(grads: Seq[INDArray]): Unit = nextLayer.update(grads)
}
