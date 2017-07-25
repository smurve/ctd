package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray

/**
  * Created by wgiersche on 25/07/17.
  */
abstract class OutputLayer extends Layer {

  /**
    * The output layer is the only layer that does not have a next layer, obviously
    * @return
    */
  override def nextLayer: Layer = throw new NoSuchMethodException("Don't call nextLayer within an Output layer!")

  override def ffwd(x: INDArray): INDArray = x

  /**
    * pass-through function to comply to layer
    * @param y the outcome of the previous layer
    * @return just what's come in
    */
  override def fun(y: INDArray): INDArray = y

  /**
    * cost/loss function
    * @param y the output of the previous layer
    * @param y_bar the expected output
    * @return the cost at the given output y
    */
  def c(y: INDArray, y_bar: INDArray ): Double

  def grad_c(x: INDArray, y_bar: INDArray): INDArray

  override def |:| ( outputLayer: OutputLayer ): Layer = {
    throw new UnsupportedOperationException("Output layer has no next layer.")
  }

  override def fwbw(x: INDArray, y_bar: INDArray): PROPAGATED = (grad_c(x, y_bar), Nil, c(x, y_bar))
}
