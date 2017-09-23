package org.smurve.shapes

import java.io.File

import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ParamAndGradientIterationListener
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.dataset.DataSet
import org.nd4s.Implicits._
import org.smurve.cifar10.Conv3ModelFactory
import org.smurve.dl4j.ActivationChecker
import org.smurve.nd4s._

/**
  * A computationally affordable demo case to check convolutional network architectures:
  * Your network should be able to classify certain shapes somewhere in the noisy image
  *
  * You can adapt the difficulty by tuning the number of classes, channels, the image size and the background noise
  */
object ShapeRunner {

  trait Params {
    def eta: Double
    def seed: Int
    def n_channels: Int
    def n_symbols: Int
    def imgSize: Int
    def noise: Double
    def n_epochs: Int
    def size_batch: Int
    def n_batches: Int
    def nf1: Int
    def nf2: Int
    def nf3: Int
    def n_dense: Int
  }


  class HardParams extends Params{
    val eta = 1e-3
    val seed = 1234
    val n_channels = 3
    val n_symbols: Int = 10
    val imgSize = 40
    val noise = 0.3
    val n_epochs = 3
    val size_batch = 500
    val n_batches = 10
    val nf1 = 30
    val nf2 = 30
    val nf3 = 100
    val n_dense = 500
  }

  class FastParams {
    val eta = 1e-3
    val seed = 1234
    val n_channels = 3
    val n_symbols: Int = 10
    val imgSize = 32
    val noise = 0.3
    val n_epochs = 2
    val batchSize = 200
    val n_batches = 10
    val nf1 = 32
    val nf2 = 64
    val nf3 = 128
    val n_dense = 1024
  }


  def main(args: Array[String]): Unit = {

    DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF)

    val defaultArgs = CmdLineArgs(n_batches = 10, size_batch = 100, nf1 = 32)

    val commandlineArgs = new CmdLineTool().parser.parse(args, defaultArgs)
      .getOrElse( throw new IllegalArgumentException)


    new HardParams {

    //new FastParams {

      override val size_batch: Int = commandlineArgs.size_batch
      override val n_batches: Int = commandlineArgs.n_batches

      val pgil = new ParamAndGradientIterationListener(
        /*iterations=*/ size_batch,
        /* printHeader = */ true,
        /*printMean=*/ false,
        /* printMinMax=*/ true,
        /*printMeanAbsValue=*/ false,
        /*outputToConsole=*/ false,
        /*outputToFile = */ true,
        /*outputToLogger = */ false,
        /*file = */ new File("shapeDataStats.csv"),
        /*delimiter = */ ",")
      val shapeData = new ShapeData(n_symbols = n_symbols, noise = noise, seed = seed)


      /* */
      private val model = new Conv3ModelFactory(
        n_classes = n_symbols, width = imgSize, height = imgSize, depth = n_channels,
        nf_1 = nf1, nf_2 = nf2, nf_3=nf3 ,n_dense = n_dense, eta = eta, seed = seed).createModel(n_channels)
      // * /
      /*
      private val model = new Conv2Model(
        n_classes = n_symbols, width = imgSize, height = imgSize, depth = n_channels,
        n_features_1 = nf1, n_features_2 = nf2, n_dense = n_dense, eta = eta, seed = seed).model
      /* */
      new SimpleCNN(n_channels, n_symbols, seed).createModel(imgSize)
      // */

      model.setListeners(/*new ScoreIterationListener(1),*/ pgil)

      println("Starting training...")

      private val trainingData = (0 until n_batches).map(_ => {
        val batch = shapeData.createBatch(imgSize, size_batch, n_channels)
        new ExistingDataSetIterator(new DataSet(batch._1, batch._2))
      })

      private val testImgs = shapeData.createBatch(imgSize, size_batch, n_channels)
      val testData = new ExistingDataSetIterator(new DataSet(testImgs._1, testImgs._2))

      println("Creating Sample data")
      private val probe = shapeData.createSamples(imgSize = imgSize, depth = n_channels, posx = 12, posy = 12)
      println("done.")

      val checker = new ActivationChecker(probe, n_channels = n_channels, imgSize = imgSize)


      /* that brought scylla down
      private val wrapper = new ParallelWrapper.Builder(model)
          .prefetchBuffer(24)
          .workers(2)
        .averagingFrequency(3)
        .reportScoreAfterAveraging(true)
        .build()
      */

      for (e <- 1 to n_epochs) {
        println(s"\nStarting epoch Nr. $e")

        for (b <- 0 until n_batches) {

          val startAt = System.currentTimeMillis()

          println(s"Epoch Nr. $e, batch Nr. ${b + 1}")

          model.fit(trainingData(b))
          val finishAt = System.currentTimeMillis()

          val eval = model.evaluate(testData)

          println(eval.stats)
          println(s"$size_batch samples learned after ${((finishAt-startAt) / 100) / 10.0} seconds.")


          testData.reset()
          trainingData(b).reset()

          //checker.analyseOutput ( model, untilLayer = 3 )
        }
      }


      /** a little sample to demonstrate inference */
      inferenceDemo(model, shapeData, 10)


      /** a little insight into the model parameters */
      println("Convolutional Layer (0):")
      printParams(model, "0_W", "0_b")

      println("Done.")


      /**
        * identify parameters by their key. This is Nd4j-specific:
        * The following key works for conv and dense layers: index _ [W|b],
        * e.g. "0_W" for the weight matrix of the very first layer
        *
        * @param model the model to be dissected
        * @param keys  key identifier for the parameters
        */
      def printParams(model: MultiLayerNetwork, keys: String*): Unit = {
        keys.foreach { key =>
          val paramVector = model.getParam(key)
          println(key)
          println(paramVector)
        }
      }


      /**
        * Demonstrate inference with a couple of newly-generated records
        *
        * @param model       the model to use
        * @param shapeData   the data generator
        * @param num_records the number of records to classify
        */
      def inferenceDemo(model: MultiLayerNetwork, shapeData: ShapeData, num_records: Int): Unit = {
        val testSet = shapeData.createBatch(imgSize, num_records, n_channels)._1
        (0 until 10).foreach { i =>

          val image = (0 until n_channels).map(c=>testSet(i, c, ->)).reduce(_ + _)

          println(visualize(image))

          val input = testSet(i, ->).reshape(1, n_channels, imgSize, imgSize)
          val prediction = model.output(input)
          println(prediction)
        }

      }
    }
  }


}
