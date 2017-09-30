package org.smurve.cifar10

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.smurve.cifar10.runner.HyperParams


class ConvolutionModelFactory(n_classes: Int = 10, width: Int = 32, height: Int = 32, depth: Int = 3,
                              nf_1: Int = 32, nf_2: Int = 64, nf_3: Int = 128, n_dense: Int = 1024,
                              eta: Double, seed: Int = 5432) {

  def this(hyperParams: HyperParams) = this(
    eta = hyperParams.eta,
    nf_1 = hyperParams.nf1,
    nf_2 = hyperParams.nf2,
    nf_3 = hyperParams.nf3,
    n_dense = hyperParams.dense)

  var index = 0

  def next: Int = {
    index += 1
    index - 1
  }

  def createModel(depth: Int): MultiLayerNetwork = {

    import Activation._
    import LossFunctions.LossFunction._
    import SubsamplingLayer._

      val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .seed(seed)

      .regularization(true).l2(0.0005)
      .learningRate(eta) //.000003
      .lrPolicyDecayRate(1e-6)
      .weightInit(WeightInit.XAVIER_UNIFORM)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(Updater.ADAM)

      .list()

      .layer(next, new ConvolutionLayer.Builder(3, 3)
        .nIn(depth)
        .stride(1, 1)
        .activation(RELU)
        .nOut(nf_1)
        .build())

      /*
      .layer(next, new ConvolutionLayer.Builder(3, 3)
        .stride(1, 1)
        .activation(RELU)
        .nOut(nf_2)
        .build())
      // */

      .layer(next, new SubsamplingLayer.Builder(PoolingType.MAX)
      .kernelSize(2, 2)
      .stride(2, 2)
      .build())

      .layer(next, new ConvolutionLayer.Builder(3, 3)
        .stride(1, 1)
        .nOut(nf_3)
        .activation(RELU)
        .build())

      .layer(next, new SubsamplingLayer.Builder(PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build())

      /*
      .layer(next, new DenseLayer.Builder()
        .activation(RELU)
        .nOut(200)
        .build())
      // */

      .layer(next, new DenseLayer.Builder()
        .activation(RELU)
        .nOut(n_dense)
        .build())

      .layer(next, new DropoutLayer.Builder(.3).build())

      .layer(next, new OutputLayer.Builder(NEGATIVELOGLIKELIHOOD)
      .nOut(n_classes)
      .activation(SOFTMAX)
      .build())

      .setInputType(InputType.convolutionalFlat(height, width, depth))
      .backprop(true).pretrain(false)

      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model
  }

}
