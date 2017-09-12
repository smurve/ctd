package org.smurve.cifar10.runner

import org.nd4j.linalg.api.buffer.DataBuffer
import scopt.OptionParser


class CmdLineTool() {

  val parser = new OptionParser[HyperParams]("CIFAR10Runner") {
    head("CIFAR10Runner", "1.0")

    opt[Int]('E', "num-epochs").valueName("Number of Epochs")
      .action((x, args) => args.copy(numEpochs = x))

    opt[Int]('N', "num-batches").valueName("Number of Batches")
      .action((x, args) => args.copy(numBatches = x))

    opt[Int]('B', "batch-size").valueName("Batch size")
      .action((x, args) => args.copy(batchSize = x))

    opt[Int]('n', "num-minibatches").valueName("Number of Minibatches")
      .action((x, args) => args.copy(numMinibatches = x))

    opt[Double]('e', "eta").valueName("Learning rate")
      .action((x, args) => args.copy(eta = x))

    opt[String]('p', "precision").valueName("numerical precision, one of 'd', 'f', 'h'")
      .action((x, args) => args.copy(precision = x))
  }


}
