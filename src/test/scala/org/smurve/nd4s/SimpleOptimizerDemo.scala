package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FlatSpec, ShouldMatchers}
import org.nd4s.Implicits._

import scala.util.Random

/**
  * Created by wgiersche on 26/07/17.
  */
class SimpleOptimizerDemo extends FlatSpec with ShouldMatchers {

  type TrainingSet = (INDArray, INDArray)

  val seed = 123L
  val rnd = new Random(seed)

  trait Setup {
    val (x1c, x2c) = (3,3)
    val N_t = 1000
    val reportEvery = 1
    val N_EPOCHS = 100
    val eta: Double = 1e-2
    val scale = 1.0

    val trSet: TrainingSet = createLabeledSet(N_t, x1c, x2c)
    val theta1: INDArray = (Nd4j.rand(seed, 3, 10) - .5) / scale
    val theta2: INDArray = (Nd4j.rand(seed, 11, 4) - .5) / scale
  }


  /**
    * provide a list of points around the given center and identify the quadrant as in
    *
    *   2 | 4
    *  ---+---
    *   1 | 3
    *
    * @param x1c the center x1 coord of the test grid
    * @param x2c the center x2 coord of the test grid
    * @return a pair consisting of the sample matrix and the labels matrix
    */
  def createLabeledSet(n: Int, x1c: Double, x2c: Double): (INDArray, INDArray) = {

    val samples = Nd4j.create(n, 2)
    val labels = Nd4j.create(n, 4)
    ( 0 until n ).foreach{ i =>
      val x1 = rnd.nextDouble * 10 - 5 + x1c
      val x2 = rnd.nextDouble * 10 - 5 + x2c
      val q = (if ( x1 > x1c ) 3 else 1) + (if ( x2 > x2c ) 1 else 0)
      val y_bar = appf(vec(1,2,3,4), d=>if(d==q) 1 else  0)
      samples(i) = vec(x1, x2)
      labels(i) = y_bar
    }
    (samples, labels)
  }


  "" should "" in {
    new Setup {

      val nn: Layer = FCL(theta1) |:| ReLU() |:| FCL(theta2) |:| Sigmoid() |:| Euclidean()

      val wild_guess: INDArray = nn.ffwd(vec(1,1))
      val (d1, g1, c1): PROPAGATED = nn.fwbw(vec(1,1), vec(1,0,0,0))

      SimpleOptimizer.train(nn, trSet, N_EPOCHS, eta, reportEvery )

      val testSize = 20
      val testSet: (INDArray, INDArray) = createLabeledSet(testSize, 3, 3)
      val educated_guess: INDArray = nn.ffwd(testSet._1)

      for ( i <- 0 until 20 ) {
        println(s"got ${educated_guess(i, ->)}, would expect ${testSet._2(i,->)}")
      }
    }
  }
}