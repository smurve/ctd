package org.smurve.mnist.models

import org.nd4j.linalg.factory.Nd4j
import org.smurve.nd4s._
import org.nd4s.Implicits._

class TooBigConvNet extends NeuralNet {

  /**
    * Starting already with meaningful weights in the conv layer
    */
  private val theta1 = vec (
    0,0,0,1,1,1,0,0,0,-1,-1,-1,
    0,0,0,-1,-1,-1,0,0,0,1,1,1,
    0,0,0,1,0,-1,1,0,-1,1,0,-1,
    0,0,0,-1,0,1,-1,0,1,-1,0,1,
    0,0,0,1,1,0,1,0,-1,0,-1,-1,
    0,0,0,-1,-1,0,-1,0,1,0,1,1,
    0,0,0,0,1,1,-1,0,1,-1,-1,0,
    0,0,0,0,-1,-1,1,0,-1,1,1,0
  ).reshape(8,4,3) / 6.0

  private val theta2 = vec (
    -1,0,1,1,-1,-1,
    -1,0,-1,-1,1,1,
    -1,0,1,-1,1,-1,
    -1,0,-1,1,-1,1,
    -1,0,1,0,0,-1,
    -1,0,-1,0,0,1,
    -1,0,0,-1,1,0,
    -1,0,0,1,-1,0
  ).reshape(8, 3, 2) / 4.0

  /** weights for the dense layers */
  private val theta3 = (Nd4j.rand(289, 200) - 0.5 ) / 10000
  private val theta4 = (Nd4j.rand(201, 10) - 0.5) / 1000

  val shape = Shape(1,28,28)
  val conv1 = Conv(theta1, 1, 28, 28,3)
  val max1 = MaxPool(1, 2, 2)
  val conv2 = Conv(theta2, 8, 13, 13,2)
  val max2 = MaxPool(8, 2, 2)
  val relu2 = ReLU()
  val dense1 = Dense(theta3)
  val dense2 = Dense(theta4)
  val output = Euclidean()

  override def model: Layer = shape !! conv1 !! max1 !! conv2 !! max2 !! Sigmoid() !!
    Flatten(8, 6,6) !! dense1 !! relu2 !! dense2 !! output


}
