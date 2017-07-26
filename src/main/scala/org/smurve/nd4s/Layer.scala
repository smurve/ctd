package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray

trait Layer {

  private var next: Option[Layer] = None

  /**
    * present the next layer
    *
    * @return the next layer. There MUST be one
    */
  def nextLayer: Layer = next.getOrElse(throw new IllegalStateException(
    "Network shouldn't end here. Are you missing an output layer?"))


  def inititialize(layer: Layer): Unit = {
    next = Some(layer)
  }

  /**
    * the function associated with this layer
    *
    * @param x the input vector
    * @return the function applied to the input vector
    */
  def fun(x: INDArray): INDArray


  /**
    * forward pass or inference: same for all layers
    *
    * @param x the batch of input row vectors
    * @return the associated output vector
    */
  def ffwd(x: INDArray): INDArray = {
    val y = fun(x)
    val value = nextLayer.ffwd(y)
    value
  }

  /**
    * forward pass and back propagation in one method call
    *
    * @param x     the batch of input row vectors
    * @param y_bar the batch of expected outcome row vectors
    */
  def fwbw(x: INDArray, y_bar: INDArray): PROPAGATED

  def update (grads: List[INDArray]): Unit

  /**
    * Stacking operator with output.
    *
    * @param outputLayer an output layer
    * @return this
    */
  def |:|(outputLayer: OutputLayer): Layer = {
    rightmost.inititialize(outputLayer)
    this
  }

  /**
    * @return the rightmost layer in the network
    */
  def rightmost: Layer = next match {
    case Some(n) => n.rightmost
    case None => this
  }

  def |:|(next: Layer): Layer = {
    val r = rightmost
    r.inititialize(next)
    this
  }
}