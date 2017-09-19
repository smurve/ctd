package org.smurve.cifar10.runner

case class HyperParams
(
  numEpochs: Int,
  numTraining: Int,
  numTest: Int,
  minibatchSize: Int,
  eta: Double,
  decay: Double,
  precision: String
)
