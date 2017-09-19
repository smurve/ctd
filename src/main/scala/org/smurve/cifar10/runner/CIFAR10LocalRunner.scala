package org.smurve.cifar10.runner

import org.deeplearning4j.datasets.iterator.ExistingDataSetIterator
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.smurve.cifar10.{CIFAR10DataReader, CIFAR10LocalContext, JoelsModel}
import org.smurve.iter.{DataSetIteratorFactory, SimpleCIFAR10BatchIterator}
import org.smurve.util.prettyPrint

import scala.language.postfixOps

object CIFAR10LocalRunner {

  def NUM_TESTS = 10000
  def NUM_RECS_PER_FILE = 10000

  def main(args: Array[String]): Unit = {

    val hyperParams = determineHyperParams ( args, defaults = HyperParams(
      numTraining = 10000,
      numTest = 10000,
      numEpochs = 5,
      minibatchSize = 100,
      eta = 1e-3,
      decay = 1e-6,
      precision = "f"
    ))

    DataTypeUtil.setDTypeForContext(dataBufferTypeFor(hyperParams.precision))

    println("Running CIFAR-10 in local mode.")
    report (hyperParams)

    println("Creating the model...")
    val model = new JoelsModel(hyperParams)
    println("Done.")

    println("Reading data from HDFS...")
    //val (trainingData, testData) = fromHdfs1( model, hyperParams )
    val (trainingData, testData) = fromHdfs( hyperParams )
    println("Done.")

    println("Starting training...")
    model.train(trainingData, testData)
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


  def fromHdfs( hyperParams: HyperParams): (DataSetIterator, DataSetIterator) = {

    val (train, test) = CIFAR10DataReader.read()

    val trainIter = new SimpleCIFAR10BatchIterator(train, hyperParams.minibatchSize)
    val testIter = new SimpleCIFAR10BatchIterator(test, hyperParams.minibatchSize)

    (trainIter, testIter)
  }

  /**
    * read the data iterators from HDFS
    */
  def fromHdfs1(model: JoelsModel, hyperParams: HyperParams): (DataSetIterator, DataSetIterator)= {


    val context = new CIFAR10LocalContext("hdfs")
    val fileCallBack = new CIFAR10SparkReader(context, hyperParams.minibatchSize)
    val factory = new DataSetIteratorFactory("hdfs://daphnis/users/wgiersche/input/cifar-10", fileCallBack)
    val numFiles = hyperParams.numTraining / NUM_RECS_PER_FILE
    val fileNames = (1 to numFiles).map(n => s"data_batch_$n.bin").toArray

    // test in batches of 100
    val (testSamples, testLabels) = context.read("test_batch.bin")
    val tes = testSamples.reshape(NUM_TESTS, 3, 32, 32)
    val tel = testLabels.reshape(NUM_TESTS, 10)
    val test = new ExistingDataSetIterator(new DataSet(tes, tel))

    //val batchReporter = new BatchReporter(model, test)

    val training = factory.createIterator(
      None, //Some(batchReporter),
      fileNames,
      numEpochs = hyperParams.numEpochs,
      chunkSize = NUM_RECS_PER_FILE,
      minibatchSize = hyperParams.minibatchSize)


    (training, test)
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
