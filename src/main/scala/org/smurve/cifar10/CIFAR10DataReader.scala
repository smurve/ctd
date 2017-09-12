package org.smurve.cifar10

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.smurve.iter.SimpleCIFAR10BatchIterator

import scala.language.postfixOps

object CIFAR10DataReader {

  def NUM_RECS = 10000
  def NUM_FILES = 5

  def main(args: Array[String]): Unit = {
    val (tr, te) = read()
    println(s"train samples: ${tr.getFeatureMatrix.shape().toList}")
    println(s"train labels : ${tr.getLabels.shape().toList}")
    println(s"test  samples: ${te.getFeatureMatrix.shape().toList}")
    println(s"test  labels : ${te.getLabels.shape().toList}")

    val iterator = new SimpleCIFAR10BatchIterator(tr, 100)
    var i = 1
    while (iterator.hasNext) {
      iterator.next()
      i += 1
    }

    println()
    println(s"Number of batches: $i")
    iterator.reset()
    println(s"Each batch: ${iterator.next().getFeatureMatrix.shape.toList}")
    println("Done.")
  }

  /**
    * read the entire training set and test set in one go. Return both sets as a pair
    * @return
    */
  def read(): (DataSet, DataSet) = {

    val context = new CIFAR10LocalContext("hdfs")
    val trainingNames = (1 to 5).map(n => s"data_batch_$n.bin").toArray

    val data: Array[(INDArray, INDArray)] = trainingNames.map(name => {
      println(s"reading $name")
      context.read(name)
    })
    println("reading test_batch.bin")
    val (teb, tel) = context.read("test_batch.bin")


    val trb = Nd4j.vstack(data.map(_._1): _*)
    val trl = Nd4j.vstack(data.map(_._2): _*)

    (new DataSet(trb, trl), new DataSet(teb, tel))
  }


}
