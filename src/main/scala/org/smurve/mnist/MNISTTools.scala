package org.smurve.mnist

import java.io.File

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

trait MNISTTools {

  def saveModel(name: String, weights: Map[String, INDArray] ) : Unit = {

    println(s"Saving model as $name")
    for ( theta <- weights ) {
      Nd4j.saveBinary(theta._2, new File ( s"${name}_${theta._1}"))
    }
  }

  def readModel(name: String, weightNames: List[String]): Map[String, INDArray] = {
    val res = for (wn <- weightNames) yield {
      val weights = Nd4j.readBinary( new File(s"${name}_$wn"))
      wn->weights
    }
    res.toMap
  }

}
