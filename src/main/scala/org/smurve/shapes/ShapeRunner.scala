package org.smurve.shapes

import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.dataset.DataSet
import org.nd4s.Implicits._
import org.smurve.nd4s._

/**
  * A computationally affordable demo case to check convolutional network architectures:
  * Your network should be able to classify certain shapes somewhere in the noisy image
  *
  * You can adapt the difficulty by tuning the number of classes, channels, the image size and the background noise
  */
object ShapeRunner {

  val seed = 1234

  val n_channels = 3
  val n_symbols: Int = 6
  val imgSize = 12 // width and height


  def main(args: Array[String]): Unit = {

    val batchSize = 2000
    val n_batches = 50

    val shapeData = new ShapeData(n_symbols = 6, noise = 0.08)

    val model = new SimpleCNN(n_channels, n_symbols, seed).createModel(imgSize)


    /** we're generating new data for each batch in a kind of online learning fashion */
    for (_ <- 0 until n_batches) {

      val batch = shapeData.createBatch(imgSize, batchSize, n_channels)
      val trainingData = new ExistingDataSetIterator(new DataSet(batch._1, batch._2))
      val testData = new ExistingDataSetIterator(new DataSet(batch._1, batch._2))

      model.fit(trainingData)
      val eval = model.evaluate(testData)

      println(eval.stats)

      testData.reset()
      trainingData.reset()
    }

    /** a little sample to demonstrate inference */
    inferenceDemo(model, shapeData, 10)


    /** a little insight into the model parameters */
    println("Convolutional Layer (0):")
    printParams(model, "0_W", "0_b")

    println("Done.")

  }



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

      val image = if (n_channels == 3)
        testSet(i, 0, ->) + testSet(i, 1, ->) + testSet(i, 2, ->)
      else if (n_channels == 2)
        testSet(i, 0, ->) + testSet(i, 1, ->)
      else
        testSet(i, 0, ->)

      println(visualize(image))

      val input = testSet(i, ->).reshape(1, n_channels, imgSize, imgSize)
      val prediction = model.output(input)
      println(prediction)
    }

  }

}
