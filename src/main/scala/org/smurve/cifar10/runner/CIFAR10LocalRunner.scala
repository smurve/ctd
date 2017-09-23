package org.smurve.cifar10.runner

import org.deeplearning4j.api.storage.impl.RemoteUIStatsStorageRouter
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.parallelism.ParallelWrapper
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration
import org.nd4j.linalg.api.memory.enums.{AllocationPolicy, LearningPolicy}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.{DataSet, MiniBatchFileDataSetIterator}
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.smurve.cifar10.{CIFAR10DataReader, Conv3ModelFactory, DataContext}
import org.smurve.iter.{SimpleCIFAR10BatchIterator, SplitBasedCIFAR10BatchIterator}
import org.smurve.util.prettyPrint

import scala.language.postfixOps

object CIFAR10LocalRunner {

  def NUM_TESTS = 10000

  def NUM_RECS_PER_FILE = 10000

  def main(args: Array[String]): Unit = {

    val hyperParams = determineHyperParams(args, defaults = HyperParams(
      parallel = 2,
      numFiles = 3,
      numTest = 1000,
      numEpochs = 4,
      minibatchSize = 100,
      eta = 2e-1,
      decay = 1e-5,
      precision = "f",
      nf1 = 20,
      nf2 = 40,
      nf3 = 100,
      dense = 500
    ))

    /**
      * 70% after 2 Epochs
      *
      * val hyperParams = determineHyperParams(args, defaults = HyperParams(
      * parallel = 1,
      * numFiles = 5,
      * numTest = 1000,
      * numEpochs = 20,
      * minibatchSize = 100,
      * eta = 5e-2,
      * decay = 1e-8,
      * precision = "d",
      * nf1 = 32,
      * nf2 = 64,
      * nf3 = 128,
      * dense = 1000
      * ))
      */


    DataTypeUtil.setDTypeForContext(dataBufferTypeFor("f"))

    println("Running CIFAR-10 in local mode.")
    report(hyperParams)


    println("Creating the model...")
    val model = new Conv3ModelFactory(hyperParams).createModel(DataContext.NUM_CHANNELS)

    val wrapper = new ParallelWrapper.Builder(model)
        .workers(hyperParams.parallel)
        .averagingFrequency(1)
        .reportScoreAfterAveraging(true)
        .build()

    println("Done.")

    println("Network Configuration:")
    println(model.getLayerWiseConfigurations.toString)
    println()

    println("Setting up UI...")
    val ui = setupUI(model)
    println("Done.")

    println("Reading data from File...")
    val (trainingIterators, testData) = fromFiles(hyperParams, nFiles = hyperParams.numFiles)
    println("Done.")

    println("Starting training...")
    (1 to hyperParams.numEpochs).foreach { epoch =>
      /*
      val learningConfig = WorkspaceConfiguration.builder
        .policyAllocation(AllocationPolicy.STRICT)
        .policyLearning(LearningPolicy.OVER_TIME)
        .build // <-- this option makes workspace learning after first loop

      Nd4j.getWorkspaceManager.getAndActivateWorkspace(learningConfig, "EPOCH")
      // */

      println(s"Epoch: $epoch")
      for (fileNum <- trainingIterators.indices) {
        println(s"Training from File Nr. $fileNum")

        wrapper.fit(trainingIterators(fileNum))

        trainingIterators(fileNum).reset()
        val evaluation = model.evaluate(testData)
        testData.reset()
        println(evaluation.stats)
      }
    }

    //ui.stop()
    println("Done.")
  }

  def setupUI(model: MultiLayerNetwork): UIServer = {
    import org.deeplearning4j.ui.api.UIServer
    import org.deeplearning4j.ui.stats.StatsListener
    //Initialize the user interface backend//Initialize the user interface backend

    val uiServer = UIServer.getInstance

    //uiServer.enableRemoteListener()

    //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
    val statsStorage = new InMemoryStatsStorage //Alternative: new FileStatsStorage(File), for saving and loading later

    //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
    uiServer.attach(statsStorage)

    //Then add the StatsListener to collect this information from the network, as it trains
    model.setListeners(new StatsListener(statsStorage))

    //val remoteUIRouter = new RemoteUIStatsStorageRouter("http://localhost:9000")


    sys.ShutdownHookThread {
      println("Shutting down the UI Server...")
      try {
        uiServer.stop()
      } catch {
        case e: Exception =>
          println(s"caught: $e. Ignoring")
      }
      println("UI Server down.")
    }

    uiServer
  }


  /**
    * determine hyperparams from defaults and command line params
    */
  def determineHyperParams(args: Array[String], defaults: HyperParams): HyperParams = {

    new CmdLineTool().parser.parse(args, defaults)
      .getOrElse({
        System.exit(-1)
        // Actually, there should be a method in scala that returns Nothing, but nobody ever cared, as it appears
        throw new RuntimeException("Just satisfying the compiler. This won't ever happen.")
      })
  }


  def fromFiles(hyperParams: HyperParams, nFiles: Int): (Array[DataSetIterator], DataSetIterator) = {

    val trainingNames = (1 to nFiles).map(n => s"data_batch_$n.bin").toArray

    val trainIters = trainingNames.map(triterator("input/cifar10/", _, hyperParams.minibatchSize))
    //new SplitBasedCIFAR10BatchIterator("input/cifar10", trainingNames, hyperParams.minibatchSize)
    val testIter = new SplitBasedCIFAR10BatchIterator("input/cifar10", Array("test_batch.bin"), hyperParams.minibatchSize)

    (trainIters, testIter)
  }


  def triterator(basePath: String, name: String, batchSize: Int): DataSetIterator = {

    val (imgs, lbls) = new DataContext(basePath).readSplit(name)

    val dataSet = new DataSet(imgs, lbls)

    new MiniBatchFileDataSetIterator(dataSet, batchSize)
  }

  /**
    * determine data type to be used
    */
  def dataBufferTypeFor(dType: String): DataBuffer.Type = {
    dType match {
      case "f" => DataBuffer.Type.FLOAT
      case "h" => DataBuffer.Type.HALF
      case _ => DataBuffer.Type.DOUBLE
    }
  }

  /**
    * report hyper parameters
    */
  def report(params: HyperParams): Unit = {
    println(s"Using ${prettyPrint(params)}")
  }


}
