package org.smurve.cifar10

import scala.language.postfixOps

object CIFAR10LocalRunner {

  val size_batch = 100
  val n_batches = 100
  val n_epochs = 30

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
    //val model = new Conv2Model(n_features_1 = 32, n_features_2 = 32, n_dense = 500, eta = 1e-4, seed = 1234)
    val model = new JoelsModel(nf_1 = 32, nf_2 = 64, nf_3=128, n_dense = 1024, eta = 1e-4, seed = 1234)
    //val model = new DenseModel(n_dense = 500, eta = 1e-3, seed = 1234)

    println("Training the model...")
    model.train(LabeledData((trs, trl), (tes, tel)), n_epochs, n_batches, size_batch)

    println("Done.")


    //model.save()
  }
}
