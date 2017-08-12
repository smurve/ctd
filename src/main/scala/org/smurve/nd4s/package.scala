package org.smurve

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.smurve.transform.{Affine, Grid, Rotate}

import scala.collection.mutable
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
  def shuffle(labeledData: (INDArray, INDArray), random: Random = new Random, transform: Affine = Affine.identity): (INDArray, INDArray) = {

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
      val j = random.nextInt(sampleSize)

      copy(samples.getRow(i), swapSample )
      copy(samples.getRow(j), samples.getRow(i))
      copy(swapSample, samples.getRow(j))

      copy(labels.getRow(i), swapLabel )
      copy(labels.getRow(j), labels.getRow(i))
      copy(swapLabel, labels.getRow(j))
    })

    if ( transform != Affine.identity ) {
      print("Starting to perturb...")
      (0 until sampleSize by 10).foreach(i => {

        if (i % 6 == 0) print(".")

        tick("copying")
        copy(samples.getRow(i), swapSample)
        tick("reshaping")
        val res = swapSample.reshape(28, 28)
        tick("rotating")
        val transformed = transform(new Grid(res))
        tick("flattening")
        val flat = Nd4j.toFlattened(transformed.field)
        tick("copying back")
        copy(flat, samples.getRow(i))
      })
    }

    println("Done.")

    bucket.foreach(entry => {
      val k = entry._1
      val s = entry._2._2
      println(s"$k: $s")
    })

    labeledData
  }

  private def tick(key: String ): Unit = {
    val t = System.currentTimeMillis()
    val entry = bucket.get(key)
    if (entry.isDefined) {
      val last = entry.get._1
      val sum = entry.get._2
      val dt = t - last
      bucket.put(key, (t, sum + dt))
    } else {
      bucket.put(key, (t, 0))
    }
  }

  private val bucket = mutable.HashMap[String, (Long, Long)]()

  def toArray (iNDArray: INDArray, width: Int = 28, height: Int = 28): Array[Double] = {
    val length = iNDArray.length
    val array = Array.fill(length){0.0}

    array.indices.foreach { index =>
      array(index) = iNDArray.getDouble(index)
    }
    array
  }

  def equiv(classification: INDArray, label: INDArray): Boolean = {
    val max = classification.max(1).getDouble(0)
    val lbl_icx = (label ** vec(0,1,2,3,4,5,6,7,8,9).T).getDouble(0).toInt
    val res = max == classification.getDouble(lbl_icx)
    res
  }



}
