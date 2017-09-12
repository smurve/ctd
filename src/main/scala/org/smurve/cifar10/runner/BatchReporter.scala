package org.smurve.cifar10.runner

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.smurve.cifar10.JoelsModel

class BatchReporter ( model: JoelsModel, testData: DataSetIterator) {

  def report (): Unit = {

    val evaluation = model.model.evaluate(testData)

    println(evaluation)
  }

}
