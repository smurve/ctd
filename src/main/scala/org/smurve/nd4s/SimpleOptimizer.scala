package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._

/**
  * Created by wgiersche on 26/07/17.
  */
object SimpleOptimizer {

  def train ( model: Layer,
              trainingSet: (INDArray, INDArray), n_epochs: Int, eta: Double, reportEvery: Int ): Unit = {

    val ( samples, labels ) = trainingSet
    for ( i <- 0 to n_epochs ) {
      val (d, g, c): PROPAGATED = model.fwbw(samples, labels)
      if ( i % reportEvery == 0 )
        println(s"Cost: $c")
      model.update(g.map(_ * -eta))
    }

  }
}
