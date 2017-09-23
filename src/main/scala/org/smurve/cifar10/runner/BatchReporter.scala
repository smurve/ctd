package org.smurve.cifar10.runner

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.smurve.cifar10.Conv3ModelFactory

class BatchReporter (model: Conv3ModelFactory, testData: DataSetIterator) {

  def report (): Unit = {

    val evaluation = model.createModel(depth = 3).evaluate(testData)

    println(evaluation)
  }

}
