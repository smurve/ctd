package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.smurve.transform.Affine

import scala.collection.immutable.Seq
import scala.util.Random

/**
  * A simple low-level SGD optimizer
  * @param generator a function that returns Affine transformations to be applied before each epoch
  * @param random the random to be used in shuffle and perturbation operations
  */
class SimpleOptimizer(val generator: () => Affine,
                      random: Random) {


  /** summing grad and cost at once. Used in reduce phase */
  def sum(a: (Seq[INDArray], Double), b: (Seq[INDArray], Double)): (Seq[INDArray], Double) = {
    val sumGrad = (a._1 zip b._1) map { case (gradA, gradB) => gradA + gradB }
    val sumCost = a._2 + b._2
    (sumGrad, sumCost)
  }

  /**
    * slice a block from the samples. Note that no data is copied here.
    * @param blockSize the number of images in the block
    * @param offset the offset to start from
    * @param block the current index
    * @param samples the INDarray of samples
    * @param labels the INDArray of labels
    * @return blocks of images and labels
    */
  def sliceBlock ( blockSize: Int, offset: Int, block: Int, samples: INDArray, labels: INDArray ): (INDArray, INDArray) = {
    val fromIndex = offset + block * blockSize
    val toIndex = fromIndex + blockSize

    ( samples(fromIndex -> toIndex, ->), labels(fromIndex -> toIndex, ->))
  }

  /**
    * Train the model using SGD
    * @param model the model to be trained
    * @param nBatches number of batches
    * @param parallelism number of parralel execution threads to use
    * @param trainingSet the training set - images and labels
    * @param testSet the test set - images and labels
    * @param n_epochs number of epochs to spend training
    * @param eta the *pesky* learning rate
    * @param reportEvery number of batches to pass before reporting cost
    */
  def train(model: Layer, nBatches: Int, parallelism: Int =1,
            trainingSet: (INDArray, INDArray), testSet: (INDArray, INDArray),
            n_epochs: Int, eta: Double, reportEvery: Int): Unit = {

    assert(parallelism >= 1, "Parallelism can't be less than 1.")

    val t0 = System.currentTimeMillis()

    val batchSize = trainingSet._1.size(0) / nBatches
    val blockSize = batchSize / parallelism
    val nBlocks = batchSize / blockSize

    for (epoch <- 1 to n_epochs) {

      println(s"Starting epoch $epoch")

      val (samples, labels) = shuffle(trainingSet, random = random, transform = generator())

      for (batchNo <- 0 until nBatches) {

        val offset = batchNo * batchSize
        val blocks = (0 until nBlocks).par

        val (g_total, c_total): (Seq[INDArray], Double) = blocks.map(block => {
            val (sample_block, label_block) = sliceBlock(blockSize, offset, block, samples, labels)
            val (_, grads, c) = model.fwbw(sample_block, label_block)
            (grads, c)
          }).reduce(sum)

        model.update(g_total.map(_ * -eta))

        if (batchNo % reportEvery == 0)
          println(s"Cost: $c_total")
      }

      val N_TEST = testSet._2.size(0)
      val res = model.ffwd(testSet._1)

      val success = ( 0 until N_TEST).map( i=>{
        val pred = res(i,->)
        val label = testSet._2(i, ->)
        if ( equiv( pred, label) ) 1.0 else 0.0
      }).sum / N_TEST

      println( s"Success rate: ${(success*1000).toInt/10.0}")
    }

    val t1 = System.currentTimeMillis()
    println(s"finished training after ${t1 - t0} ms.")

  }


}
