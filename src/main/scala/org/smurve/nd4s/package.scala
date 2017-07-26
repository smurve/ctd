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


}
