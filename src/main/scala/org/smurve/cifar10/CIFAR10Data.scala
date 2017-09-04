package org.smurve.cifar10

import org.nd4j.linalg.api.ndarray.INDArray

case class CIFAR10Data(training: (INDArray, INDArray), test: (INDArray, INDArray) )
