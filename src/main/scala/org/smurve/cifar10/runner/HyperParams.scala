package org.smurve.cifar10.runner

case class HyperParams
(
  numEpochs: Int,
  numBatches: Int,
  batchSize: Int,
  numMinibatches: Int,
  eta: Double,
  decay: Double,
  precision: String
)
