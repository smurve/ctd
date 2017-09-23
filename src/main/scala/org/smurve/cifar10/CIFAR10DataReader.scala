package org.smurve.cifar10

import org.nd4j.linalg.dataset.DataSet

import scala.language.postfixOps

object CIFAR10DataReader {

  /**
    * read the entire training set and test set in one go. Return both sets as a pair
    * @return
    */
  def read(n: Int): (Array[DataSet], DataSet) = {

    val context = new DataContext("./input/cifar10")
    val trainingNames = (1 to n).map(n => s"data_batch_$n.bin").toArray

    val trainingData: Array[DataSet] = trainingNames.map(name => {
      println(s"reading $name")
      context.read(name)
    }).map(p=>new DataSet(p._1, p._2))

    println("reading test_batch.bin")
    val (teb, tel) = context.read("test_batch.bin")


    (trainingData, new DataSet(teb, tel))
  }


}
