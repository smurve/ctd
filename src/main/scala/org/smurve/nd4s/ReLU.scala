package org.smurve.nd4s
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms._

/**
  * Created by wgiersche on 26/07/17.
  */
case class ReLU() extends Activation {
  /**
    * the function associated with this layer
    *
    * @param x the input vector
    * @return the function applied to the input vector
    */
  override def fun(x: INDArray): INDArray = relu(x) //x.map(d=>if(d>0) d else 0)

  def f_prime (x: Double): Double = if(x>0) 1 else 0

  /**
    * forward pass and back propagation in one method call
    *
    * @param x     the batch of input row vectors
    * @param y_bar the batch of expected outcome row vectors
    */
  override def fwbw(x: INDArray, y_bar: INDArray): (INDArray, List[INDArray], Double) = {
    val (dC_dy, grads, c) = nextLayer.fwbw(fun(x), y_bar)
    (dC_dy * f_prime(x), grads, c)
  }
}
