package org.smurve.cifar10.runner

import java.io.File

import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.smurve.iter.SplitBasedCIFAR10BatchIterator
import org.smurve.cifar10._
import org.smurve.nd4s._
import org.nd4s.Implicits._

object CIFAR10Inferencer {

  def main(args: Array[String]): Unit = {
    val savedModel = new File("CIFAR10-Model.zip")
    val model = ModelSerializer.restoreMultiLayerNetwork(savedModel)

    var testIter = new SplitBasedCIFAR10BatchIterator("input/cifar10", Array("test_batch.bin"),1)

    for ( _ <- 0 to 10) {
      val ds = testIter.next
      val inp = ds.getFeatureMatrix
      val classd = model.output(inp)
      val img = asImage(inp.reshape(3,32,32))
      val trueLabel = (ds.getLabels ** vec(0,1,2,3,4,5,6,7,8,9).T).getInt(0)
      val guess = understand(classd)
      val l = categories(trueLabel)

      // best two guesses
      val b1 = categories(guess.head._2)
      val b1_confidence = guess.head._1
      val b2 = categories(guess(1)._2)
      val b2_confidence = guess(1)._1

      val fileName = s"output/$l=${b1}_or_$b2.png"
      img.output(fileName)
    }
  }

  def understand(inda: INDArray): Seq[(Double, Int)] = {
    val res = new Array[Double](10)
    for ( i <- 0 to 9 ) {
      res(i) = inda.getDouble(i)
    }
    res.toList.zipWithIndex.sortBy(-_._1)
  }
}
