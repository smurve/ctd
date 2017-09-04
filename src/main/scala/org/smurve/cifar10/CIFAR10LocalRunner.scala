package org.smurve.cifar10

import scala.language.postfixOps

object CIFAR10LocalRunner {

  val size_batch = 1000
  val n_batches = 10
  val n_epochs = 50

  def main(args: Array[String]): Unit = {

    val context = new CIFAR10LocalContext("hdfs")

    println("Reading training data from hdfs...")
    val (trainingSamples, trainingLabels) = context.read("data_batch_1.bin")
    println("Done.")
    val trs = trainingSamples.reshape(n_batches, size_batch, 3, 32, 32)
    val trl = trainingLabels.reshape(n_batches, size_batch, 10)

    println("Reading test data from hdfs...")
    val (testSamples, testLabels) = context.read("test_batch.bin")
    val tes = testSamples.reshape(n_batches, size_batch, 3, 32, 32)
    val tel = testLabels.reshape(n_batches, size_batch, 10)

    println("Done.")

    println("Creating the model...")
    val model = new CIFAR10Model(seed = 5432).train(CIFAR10Data((trs, trl), (tes, tel)), n_epochs, n_batches, size_batch)
    println("Done.")

    //model.save()
  }
}
