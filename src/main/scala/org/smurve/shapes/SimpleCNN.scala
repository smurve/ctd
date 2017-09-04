package org.smurve.shapes

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{ConvolutionMode, MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, OutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions

/**
  * creates a simple conv net with one conv layer and one dense layer. Used to demonstrate the model checker.
  * Uses one filter per class
  * @param n_channels number of input channels, typically 3 for images
  * @param n_classes number of 4x4 symbols, ranging from 2 to 6
  * @param seed the seed for the random initialization
  */
class SimpleCNN (val n_channels: Int, val n_classes: Int, val seed: Int){

  def createModel(imageSize: Int): MultiLayerNetwork = {

    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .regularization(false) //.l2(0.0005)
      .learningRate(.001)
      .weightInit(WeightInit.XAVIER)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(Updater.NESTEROVS)

      .list()

      .layer(0, new ConvolutionLayer.Builder(4, 4)
        .convolutionMode(ConvolutionMode.Same)
        .nIn(n_channels)
        .stride(1, 1)
        .activation(Activation.SOFTPLUS)
        .nOut(n_classes)
        .build())


      .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nOut(n_classes)
        .activation(Activation.SOFTMAX)
        .build())

      .setInputType(InputType.convolutionalFlat(imageSize, imageSize, 1))
      .backprop(true).pretrain(false)

      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model
  }

}
