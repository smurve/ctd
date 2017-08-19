package org.smurve.mnist.models

import org.smurve.nd4s.Layer

trait NeuralNet {

  def model: Layer
}
