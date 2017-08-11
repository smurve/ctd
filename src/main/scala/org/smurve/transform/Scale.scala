package org.smurve.transform

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.smurve.nd4s.vec


case class Scale(x: Double = 1.0, y: Double = 1.0) extends Affine {

  require(x <= 1.2, s"x shouldn't exceed 1.2, got ${x}")
  require(x >= .8, s"x shouldn't be smaller than 0.8, got ${x}")
  require(y <= 1.2, s"y shouldn't exceed 1.2, got ${y}")
  require(y >= 0.8, s"y shouldn't be smaller than 0.8, got ${y}")

  val M_INV: INDArray = vec(1.0/x, 0, 0,  1.0/y).reshape(2,2)

  /**
    * the inverse transformation to be applied to the underlying coordinate system.
    *
    * @param x the original x as one or more row vectors
    * @return the new coordinate
    */
  override def invMap(x: INDArray): INDArray = {
    require ( x.size(1) == 2, s"Can only map N x 2 vectors (representing N 2-dim row vectors). Shape was (${x.size(0)},${x.size(1)})")
    val res = M_INV ** x.T
    res.T
  }
}
