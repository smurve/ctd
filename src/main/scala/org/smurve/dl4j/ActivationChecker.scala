package org.smurve.dl4j

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._

class ActivationChecker(probes: INDArray, n_channels: Int, imgSize: Int) {

  /**
    * display the output of all layers
    *
    * @param model the model to dissect
    */
  def analyseOutput(model: MultiLayerNetwork, untilLayer: Int): Unit = {

    def outputIterator ( iNDArray: INDArray ) = {
      val newShape = iNDArray.shape().updated(0, 1)
      (0 until iNDArray.size(0)).map(single =>iNDArray(single, ->).reshape(newShape: _*))
    }

    val test = outputIterator(probes)

    def activateForAllProbes(layer: Int): Seq[INDArray] = {
      outputIterator(probes).map { p =>
        activateForSingleProbe(layer, p)
      }
    }

    def activateForSingleProbe(layer: Int, singleProbe: INDArray): INDArray = {
      if (layer == 0)
        model.getLayer(0).activate(singleProbe, false)
      else {
        val fromPrevious = activateForSingleProbe(layer - 1, singleProbe)
        model.getLayer(layer).activate(fromPrevious, false)
      }
    }

    val res = (0 to untilLayer).map(activateForAllProbes)

    val report = createReport(untilLayer, res)
    println(report)

  }


  def createReport(layer: Int, activations: Seq[Seq[INDArray]]): String = {

    val sumOfAll = activations.flatMap(l => {
      l.map(a => {
        val max = math.abs(a.maxT[Double])
        val min = math.abs(a.minT[Double])
        min + max
      })
    }).sum


    activations.zipWithIndex.map(l => {

      l._1.zipWithIndex.map(a => {

        val probeNr = a._2
        val max = a._1.maxT[Double]
        val min = a._1.minT[Double]
        val warning = if (sumOfAll == 0.0)
          """
            |
            |
            |      ************************** WARNING!!! *****************************"
            |
            |                               TRAINING FAILURE"
            |                             "ACTIVATION VANISHED"
            |
            |      ************************** WARNING!!! *****************************"
            |""".stripMargin
        else ""

        s"max(p$probeNr)=$max, min(p$probeNr)=$min $warning"
      }).map(line=>s"Layer: ${l._2} $line").mkString(" / ")
    }).mkString("\n")

  }
}
