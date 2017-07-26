package org.smurve

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

/**
  * Created by wgiersche on 25/07/17.
  */
package object nd4s {

  /**
    * the return value of fwbw as a tuple consisting of
    * _1: dC/dX of the layer returning this = dC/dy of the receiving layer. This is the actual back prop term
    * _2: List of all dC/dTheta gradients of the subsequent layers. Prepend your grad to bevor returning from fwbw
    * _3: the current cost
    */
  type PROPAGATED = (INDArray, List[INDArray], Double )

  /**
    * vertically "pad" with ones.
    *
    *                     1  1
    *
    *        a  b         a  b
    *                =>
    *        c  d         c  d
    *
    * @param x the input
    * @return the input matrix, padded with ones
    */
  def v1(x: INDArray): INDArray = Nd4j.vstack(Nd4j.ones(x.size(1)), x)

  /**
    * horizontically "pad" with ones.
    *
    *        a  b         1  a  b
    *                =>
    *        c  d         1  c  d
    *
    * @param x the input
    * @return the input matrix, padded with ones
    */
  def h1(x: INDArray): INDArray = Nd4j.hstack(Nd4j.ones(x.size(0)).T, x)

  /**
    * convenience vector literals
    * @param arr the numbers to make up the INDArray
    * @return the INDArray containing those numbers
    */
  def vec(arr: Double*): INDArray = Nd4j.create(Array(arr: _*))


  /**
    * work-around for broken map() on INDArray
    * Only supporting tensors of rank 1 and 2
    */
  def appf(x: INDArray, f: Double=>Double ): INDArray = {
    val res = Nd4j.zeros(x.shape: _*)
    val shape1 = if (x.shape.length == 1) 1 +: x.shape else x.shape
    for ( i <- 0 until shape1(0))
      for ( j <- 0 until shape1(1))
        res(i,j) = f(x(i,j))
    res
  }
}
