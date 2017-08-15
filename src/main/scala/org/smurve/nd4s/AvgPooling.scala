package org.smurve.nd4s
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
/**
  *
  * @param depth_stride the stride depth
  * @param height_stride the vertical stride size
  * @param width_stride the horizontal stride size
  */
case class AvgPooling (depth_stride: Int, height_stride: Int, width_stride: Int) extends Layer {

  val N_values: Int = depth_stride * height_stride * width_stride

  /**
    * averaging over depth and stride, which represents the different feature map.
    * Note that we require the strides to perfectly fit into the input.
    *
    * @param x the input vector of rank 4: N_features x D x H x W
    * @return the function applied to the input vector
    */
  override def fun(x: INDArray): INDArray = {
    require(x.rank == 4, "Need to be rank 4: N_features x D x H x W")
    require ( x.size(2) % height_stride == 0, "stride height doesn't divide input height.")
    require ( x.size(3) % width_stride == 0, "stride width doesn't divide input width.")

    val res = Nd4j.zeros(x.size(0), x.size(2) / height_stride, x.size(3) / width_stride)

    for {
      n <- 0 until x.size(0)
      ir <- 0 until x.size(2) by height_stride
      ic <- 0 until x.size(3) by width_stride
    }
        res(n, ir/height_stride, ic/width_stride) = (for {
          d <- 0 until depth_stride
          r <- ir until ir + height_stride
          c <- ic until ic + width_stride
        } yield x(n, d,r,c)).sum / N_values
    res
  }

  /**
    * forward pass and back propagation in one method call
    *
    * @param x     the batch of input row vectors
    * @param y_bar the batch of expected outcome row vectors, will be passed on to the output layer
    */
  override def fwbw(x: INDArray, y_bar: INDArray): (INDArray, List[INDArray], Double) = {
    val (dC_dy, grads, c) = nextLayer.fwbw(fun(x), y_bar)
    val dC_dx = Nd4j.zeros(x.shape: _*)

    for {
      od <- 0 until dC_dy.size(0)
      or <- 0 until dC_dy.size(1) by height_stride
      oc <- 0 until dC_dy.size(2) by width_stride
      d <- 0 until depth_stride
      r <- or until or + height_stride
      c <- oc until oc + width_stride

      // chain rule again: 1/N = dy/dx
      } dC_dx(od, d, r, c) = dC_dy(od, or, oc) / N_values

    (dC_dx, grads, c)
  }


  def dy_dx(od: Int, or: Int, oc: Int, id: Int, id2: Int, ir: Int, ic: Int) = {


    ???
  }


  /**
    * No params - nothing to do here. Just forward
    * @param grads the amount to be added
    */
  override def update(grads: Seq[INDArray]): Unit = nextLayer.update(grads)
}
