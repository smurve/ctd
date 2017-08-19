package org.smurve.nd4s
import org.nd4j.linalg.api.ndarray.INDArray

case class Shape (shape: Int*) extends Layer {

  val singleSize: Int = shape.product

  /**
    * reshape the input vector
    */
  override def fun(x: INDArray): INDArray = {
    val newShape = (x.length / singleSize) +: shape.toArray
    x.reshape(newShape: _*)
  }

  /**
    * reshape and forward
    */
  override def fwbw(x: INDArray, y_bar: INDArray): (INDArray, List[INDArray], Double) = {
    val (dC_dy, grads, c ) = nextLayer.fwbw(fun(x), y_bar)
    (dC_dy.reshape(x.shape: _*), grads, c)
  }

  /**
    * simply forward
    */
  override def update(grads: Seq[INDArray]): Unit = nextLayer.update(grads)
}
