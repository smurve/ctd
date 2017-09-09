package org.smurve.cifar10

import org.nd4j.linalg.api.ndarray.INDArray

case class LabeledData(training: (INDArray, INDArray), test: (INDArray, INDArray) )
