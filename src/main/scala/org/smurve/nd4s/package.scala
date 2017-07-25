package org.smurve

import org.nd4j.linalg.api.ndarray.INDArray

/**
  * Created by wgiersche on 25/07/17.
  */
package object nd4s {

  /**
    * the return value of fwbw as a tuple consisting of
    * _1: dC/dX of the layer returning this = dC/dy of the receiving layer. This is the actual back prop term
    * _2: List of all dC/dTheta gradients of the subsequent layers. Prepend your grad to bevor returning from fwbw
    * _3:
    */
  type PROPAGATED = (INDArray, List[INDArray], Double )
}
