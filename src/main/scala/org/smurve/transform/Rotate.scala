package org.smurve.transform

import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.smurve.nd4s.vec

case class Rotate(phi: Double ) extends Affine {

  /** the inverse rotation matrix */
  val M_INV: INDArray = {
    val c = math.cos(phi)
    val s = math.sin(phi)
    vec(c, s, -s, c ).reshape(2,2)
  }

  /**
    * the inverse transformation to be applied to the underlying coordinate system.
    * @param x the original x
    * @return the new coordinate
    */
  def invMap(x: INDArray ): INDArray = {
    require ( x.size(1) == 2, "can only map N x 2 vectors (representing N 2-dim row vectors).")
    val res = M_INV ** x.T
    res.T
  }
}

