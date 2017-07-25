package org.smurve.nd4s
import org.nd4j.linalg.api.ndarray.INDArray

import org.nd4s.Implicits._

/**
  * Fully connected layer. Just needs to implement fwbw
  * @param theta the weight matrix
  */
case class FCL(theta: INDArray) extends Layer {

  def fun(x: INDArray): INDArray = x ** theta

  def fwbw(x: INDArray, y_bar: INDArray): PROPAGATED = {
    val (dC_dy, grads, cost) = nextLayer.fwbw(fun(x), y_bar)
    val dC_dx = dC_dy ** theta.T
    val grad = x.T ** dC_dy
    (dC_dx, grad :: grads, cost)
  }

  def += ( dTheta: INDArray ): Unit = theta += dTheta
}

