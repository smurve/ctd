package org.smurve.transform

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.smurve.nd4s.vec

case class Trans(x: Double = 0, y: Double = 0) extends Affine {

  val axis: INDArray = vec(-x, -y)

  /**
    * the inverse transformation to be applied to the underlying coordinate system.
    *
    * @param x the original x as one or more row vectors
    * @return the new coordinate
    */
  override def invMap(x: INDArray): INDArray = {
    require ( x.size(1) == 2, s"Can only map N x 2 vectors (representing N 2-dim row vectors). Shape was (${x.size(0)},${x.size(1)})")
    val res = axis + x
    res
  }
}
