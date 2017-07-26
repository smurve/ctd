package org.smurve.nd4s
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import scala.language.postfixOps

/**
  * Fully connected layer. Just needs to implement fwbw
  * @param theta the weight matrix
  */
case class FCL(theta: INDArray) extends Layer {

  def fun(x: INDArray): INDArray = h1(x) ** theta

  def fwbw(x: INDArray, y_bar: INDArray): PROPAGATED = {
    val (dC_dy, grads, cost) = nextLayer.fwbw(fun(x), y_bar)
    val dC_dx = dC_dy ** theta(1->,->).T
    val grad = h1(x).T ** dC_dy
    (dC_dx, grad :: grads, cost)
  }

  def += ( dTheta: INDArray ): Unit = theta += dTheta

  /**
    * update from head and pass the tail on to subsequent layers
    * @param grads: The list of gradients accumulated during training
    */
  override def update(grads: Seq[INDArray]): Unit = {
    this += grads.head
    nextLayer.update(grads.tail)
  }
}

