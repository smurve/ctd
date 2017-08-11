package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.smurve.transform.Affine

import scala.util.Random

/**
  * Created by wgiersche on 26/07/17.
  */
object SimpleOptimizerDemo  {

  type TrainingSet = (INDArray, INDArray)

  val seed = 128L
  val rnd = new Random(seed)

  trait Setup {
    val (x1c, x2c) = (3, 3)
    val N_t = 600
    val N_EPOCHS = 500
    val reportEvery = 20
    val eta: Double = 1e-3
    val scale = 1.0
    val nbatches = 1
    val parallel = false
    val task = false // task or data parallel

    val trSet: TrainingSet = createLabeledSet(N_t, x1c, x2c)
    val theta1: INDArray = (Nd4j.rand(seed, 3, 10) - .5) / scale
    val theta2: INDArray = (Nd4j.rand(seed, 11, 4) - .5) / scale
  }


  /**
    * provide a list of points around the given center and identify the quadrant as in
    *
    * 2 | 4
    * ---+---
    * 1 | 3
    *
    * @param x1c the center x1 coord of the test grid
    * @param x2c the center x2 coord of the test grid
    * @return a pair consisting of the sample matrix and the labels matrix
    */
  def createLabeledSet(n: Int, x1c: Double, x2c: Double): (INDArray, INDArray) = {

    val samples = Nd4j.create(n, 2)
    val labels = Nd4j.create(n, 4)
    (0 until n).foreach { i =>
      val x1 = rnd.nextDouble * 10 - 5 + x1c
      val x2 = rnd.nextDouble * 10 - 5 + x2c
      val q = (if (x1 > x1c) 3 else 1) + (if (x2 > x2c) 1 else 0)
      val y_bar = appf(vec(1, 2, 3, 4), d => if (d == q) 1 else 0)
      samples(i) = vec(x1, x2)
      labels(i) = y_bar
    }
    (samples, labels)
  }

  def matches(y: INDArray, y_bar: INDArray): Boolean = {
    val found = (0 until y.length).map(y(_))
    val given = (0 until y.length).map(y_bar(_))
    (found zip given).map(p => p._2 == 1.0 && p._1 > 0.5).reduce(_ || _)
  }

  def main(args: Array[String]): Unit = {
    new Setup {

      val nn: Layer = FCL(theta1) |:| ReLU() |:| FCL(theta2) |:| Sigmoid() |:| Euclidean()

      val wild_guess: INDArray = nn.ffwd(vec(1, 1))
      val (d1, g1, c1): PROPAGATED = nn.fwbw(vec(1, 1), vec(1, 0, 0, 0))

      val optimizer = new SimpleOptimizer(()=>Affine.identity, random = new Random(seed))
      optimizer.train(model = nn, nBatches = nbatches, parallel = parallel, task = task,
        trainingSet = trSet, n_epochs = N_EPOCHS, eta = eta, reportEvery = reportEvery)

      val testSize = 100
      val (samples, labels): (INDArray, INDArray) = createLabeledSet(testSize, 3, 3)
      val educated_guess: INDArray = nn.ffwd(samples)

      for (i <- 0 until testSize) {
        if (!matches(educated_guess(i, ->), labels(i, ->))) {
          println(s"For ${samples(i, ->)} got ${educated_guess(i, ->)}, but would expect ${labels(i, ->)}")
        }
      }

      val success_rate: Double = (0 until testSize).
        map(i => if (matches(educated_guess(i, ->), labels(i, ->))) 1 else 0).sum * 100.0 / testSize

      println(s"Success rate: $success_rate")
    }
  }
}
