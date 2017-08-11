package org.smurve.transform

import org.nd4j.linalg.api.ndarray.INDArray
import org.smurve.nd4s.vec
import org.nd4s.Implicits._

case class Shear(x: Double, y: Double) extends Affine {

  val M_INV: INDArray = vec(1, -x, -y, 1 ).reshape(2,2)

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
