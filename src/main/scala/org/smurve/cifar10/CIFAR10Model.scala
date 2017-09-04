package org.smurve.cifar10

import java.io.File

import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, DenseLayer, OutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.conf.{LearningRatePolicy, MultiLayerConfiguration, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.{ParamAndGradientIterationListener, ScoreIterationListener}
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4s.Implicits._


class CIFAR10Model(seed: Int = 5432) {

  val N_CHANNELS = 3
  val N_CLASSES = 10

  val model: MultiLayerNetwork = createModel()


  def train(data: CIFAR10Data, n_epochs: Int, n_batches: Int, size_batches: Int): Unit = {

    val pgil = new ParamAndGradientIterationListener(
      /*iterations=*/ size_batches,
      /* printHeader = */ true,
      /*printMean=*/ false,
      /* printMinMax=*/ true,
      /*printMeanAbsValue=*/ false,
      /*outputToConsole=*/ false,
      /*outputToFile = */ true,
      /*outputToLogger = */ false,
      /*file = */ new File("stats.csv"),
      /*delimiter = */ ",")

    println("Starting training...")
    model.setListeners(new ScoreIterationListener(1), pgil)

    for (epoch <- 0 until n_epochs) {
      println(s"Starting epoch Nr. $epoch")
      for (batch <- 0 until n_batches) {
        println(s"Starting batch Nr. $batch")

        val trbs = data.training._1(batch, ->, ->, ->, ->)
        //Nd4j.shuffle(trbs, 1,2,3)
        val trbl = data.training._2(batch, ->, ->)

        val tebs = data.test._1(batch, ->, ->, ->, ->)
        val tebl = data.test._2(batch, ->, ->)

        val trainingData = new ExistingDataSetIterator(new DataSet(trbs, trbl))
        val testData = new ExistingDataSetIterator(new DataSet(tebs, tebl))

        model.fit(trainingData)
        val eval = model.evaluate(testData)
        println(eval.stats)
        testData.reset()
      }
    }
    println("Done.")
  }

  def createModel(): MultiLayerNetwork = {


    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .seed(seed)
      //.regularization(true).l2(0.0005)
      .learningRate(.000001) //.000003
      .weightInit(WeightInit.XAVIER)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(Updater.NESTEROVS)
      .list()

      .layer(0, new ConvolutionLayer.Builder(5, 5)
        .nIn(N_CHANNELS)
        .stride(1, 1)
        .activation(Activation.SOFTPLUS)
        .nOut(32)
        .build())

      .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build())

      .layer(2, new ConvolutionLayer.Builder(5, 5)
        .stride(1, 1)
        .nOut(32)
        .activation(Activation.SOFTPLUS)
        .build())

      .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
        .kernelSize(2, 2)
        .stride(2, 2)
        .build())

      .layer(4, new DenseLayer.Builder()
        .activation(Activation.SOFTPLUS)
        .nOut(500)
        .build())

      .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .nOut(N_CLASSES)
        .activation(Activation.SOFTMAX)
        .build())

      .setInputType(InputType.convolutionalFlat(32, 32, 3))
      .backprop(true).pretrain(false)

      .build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model
  }

}
