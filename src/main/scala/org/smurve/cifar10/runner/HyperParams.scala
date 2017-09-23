package org.smurve.cifar10.runner

case class HyperParams
(
  parallel: Int,
  numEpochs: Int,
  numFiles: Int,
  numTest: Int,
  minibatchSize: Int,
  eta: Double,
  decay: Double,
  precision: String,
  nf1: Int,
  nf2: Int,
  nf3: Int,
  dense: Int
)
