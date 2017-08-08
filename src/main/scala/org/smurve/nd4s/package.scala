package org.smurve

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.smurve.mnist.MNISTImage

import scala.util.Random

/**
  * Created by wgiersche on 25/07/17.
  */
package object nd4s {

  /**
    * the return value of fwbw as a tuple consisting of
    * _1: dC/dX of the layer returning this = dC/dy of the receiving layer. This is the actual back prop term
    * _2: List of all dC/dTheta gradients of the subsequent layers. Prepend your grad to bevor returning from fwbw
    * _3: the current cost
    */
  type PROPAGATED = (INDArray, List[INDArray], Double )

  /**
    * vertically "pad" with ones.
    *
    *                     1  1
    *
    *        a  b         a  b
    *                =>
    *        c  d         c  d
    *
    * @param x the input
    * @return the input matrix, padded with ones
    */
  def v1(x: INDArray): INDArray = Nd4j.vstack(Nd4j.ones(x.size(1)), x)

  /**
    * horizontically "pad" with ones.
    *
    *        a  b         1  a  b
    *                =>
    *        c  d         1  c  d
    *
    * @param x the input
    * @return the input matrix, padded with ones
    */
  def h1(x: INDArray): INDArray = Nd4j.hstack(Nd4j.ones(x.size(0)).T, x)

  /**
    * convenience vector literals
    * @param arr the numbers to make up the INDArray
    * @return the INDArray containing those numbers
    */
  def vec(arr: Double*): INDArray = Nd4j.create(Array(arr: _*))


  /**
    * work-around for broken map() on INDArray
    * Only supporting tensors of rank 1 and 2
    */
  def appf(x: INDArray, f: Double=>Double ): INDArray = {
    val res = Nd4j.zeros(x.shape: _*)
    val shape1 = if (x.shape.length == 1) 1 +: x.shape else x.shape
    for ( i <- 0 until shape1(0))
      for ( j <- 0 until shape1(1))
        res(i,j) = f(x(i,j))
    res
  }

  /**
    *
    * @param labeledData the images/labels to be shuffled
    * @return
    */
  def shuffle(labeledData: (INDArray, INDArray), rnd: Random = new Random): (INDArray, INDArray) = {
    def copy ( from: INDArray, to: INDArray): Unit = {
      val l = from.length()
      (0 until l).foreach(i=>to(i)=from(i))
    }
    val ( samples, labels ) = labeledData
    val sampleSize = samples.size(1)
    val swapSample = Nd4j.zeros(sampleSize)
    val labelSize = labels.size(1)
    val swapLabel = Nd4j.zeros(labelSize)

    ( 0 until sampleSize).foreach (i=> {
      val j = rnd.nextInt(sampleSize)
      copy(samples.getRow(i), swapSample )
      copy(samples.getRow(j), samples.getRow(i))
      copy(swapSample, samples.getRow(j))
      copy(labels.getRow(i), swapLabel )
      copy(labels.getRow(j), labels.getRow(i))
      copy(swapLabel, labels.getRow(j))
    })

    labeledData
  }

  /**
    * @param iNDArray the INDArray representing the MNIST image to be printed
    * @param width the width of the image, defaults to 28
    * @param height the height of the image, defaults to 28
    * @return a printable string representation of the given image
    */
  def asString (iNDArray: INDArray, width: Int = 28, height: Int = 28): String = {
    val length = iNDArray.length
    val array = Array.fill(length){0.toByte}

    array.indices.foreach { index =>
        array(index) = iNDArray.getDouble(index).toByte
    }
    MNISTImage(array, width, height).toString
  }


}
