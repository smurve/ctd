package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

/**
  *
  * @param depth_stride  the stride depth
  * @param height_stride the vertical stride size
  * @param width_stride  the horizontal stride size
  */
case class MaxPool(depth_stride: Int, height_stride: Int, width_stride: Int) extends Layer with ParameterSupport {

  val N_values: Int = depth_stride * height_stride * width_stride

  /**
    * @return the indices that constitue the source of the output at (od, or, oc)
    */
  def domainOf(n: Int, od: Int, or: Int, oc: Int): IArr =
    iArr(
      n until n + 1,
      od until od + 1,
      0 until depth_stride,
      height_stride * or until height_stride * (or + 1),
      width_stride * oc until width_stride * (oc + 1))


  /**
    * @return the pooling function values and partial derivatives in one go.
    */
  def fun2(x: INDArray): (INDArray, INDArray) = {
    require(x.rank == 5, "Need to be rank 5: N_samples x N_features x D x H x W")
    require(x.size(3) % height_stride == 0, "stride height doesn't divide input height.")
    require(x.size(4) % width_stride == 0, "stride width doesn't divide input width.")
    val N_inp = x.size(0)

    val res = Nd4j.zeros(N_inp, x.size(1), x.size(3) / height_stride, x.size(4) / width_stride)
    val dy_dx = Nd4j.zeros(x.shape: _*)

    for (Array(n, od, or, oc) <- iArr(res)){
      // look for max at those indices contributing to output at (od, or, oc)
      val domain = domainOf(n, od, or, oc)
      val max = maxWithIndex(x, domain)
      res(n, od, or, oc) = max._1
      val i = max._2
      dy_dx(i(0), i(1), i(2), i(3), i(4)) = 1
    }

    printOutput(res)
    (res, dy_dx)
  }

  override def fun(x: INDArray): INDArray = fun2(x)._1


  /**
    * forward pass and back propagation in one method call
    *
    * @param x     the batch of input row vectors
    * @param y_bar the batch of expected outcome row vectors, will be passed on to the output layer
    */
  override def fwbw(x: INDArray, y_bar: INDArray): (INDArray, List[INDArray], Double) = {

    val (f, dy_dx) = fun2(x)
    val (dC_dy, grads, c) = nextLayer.fwbw(f, y_bar)
    val dC_dx = Nd4j.zeros(x.shape: _*)

    for {
      Array(n, od, or, oc) <- iArr(dC_dy)
      Array(n, id, d, ir, ic) <- domainOf(n, od, or, oc)
    }
      dC_dx(n, id, d, ir, ic) = dy_dx(n, id, d, ir, ic) * dC_dy(n, od, or, oc)


    (dC_dx, grads, c)
  }


  /**
    * No params - nothing to do here. Just forward
    *
    * @param grads the amount to be added
    */
  override def update(grads: Seq[INDArray]): Unit = nextLayer.update(grads)


  def numOutputVectors: Int = integerParam("print.output").getOrElse(0)

  def printOutput(array: INDArray): Unit = {
    val n = numOutputVectors
    if ( n > 0 ) {
      for (i <- 0 until n ) {
        val s = for {
          od <- 0 until array.size(1)
        } yield {
          visualize(array(i, od, ->, ->).reshape(array.size(2), array.size(3)))
        }
        println(in_a_row(" | ")(s: _*))
      }
    }
  }


}
