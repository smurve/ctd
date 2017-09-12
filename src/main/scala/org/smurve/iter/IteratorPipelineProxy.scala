package org.smurve.iter

import java.util

import org.nd4j.linalg.dataset.{DataSet, MiniBatchFileDataSetIterator}
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.smurve.cifar10.runner.BatchReporter


class IteratorPipelineProxy(incoming: DataSetIterator, numBatches: Int, batchSize: Int, numMiniBatches: Int) extends DataSetIterator {

  val outgoing: Array[DataSetIterator] = new Array[DataSetIterator](numBatches)

  outgoing(0) = {
    val first = incoming.next()
    new MiniBatchFileDataSetIterator(first, batchSize / numMiniBatches)
  }

  var reporters: List[BatchReporter] = Nil

  var currBatch = 0

  def addReporter(batchReporter: BatchReporter): Unit = {
    reporters = batchReporter :: reporters
  }

  override def cursor(): Int = currBatch * batchSize + outgoing(currBatch).cursor()

  override def next(num: Int): DataSet = throw new UnsupportedOperationException

  override def setPreProcessor(preProcessor: DataSetPreProcessor): Unit = incoming.setPreProcessor(preProcessor)

  override def getPreProcessor: DataSetPreProcessor = incoming.getPreProcessor

  override def totalOutcomes(): Int = incoming.totalOutcomes()

  override def getLabels: util.List[String] = incoming.getLabels

  override def inputColumns(): Int = incoming.inputColumns()

  override def resetSupported(): Boolean = incoming.resetSupported()

  override def asyncSupported() = false

  override def batch: Int = batchSize

  override def reset(): Unit = {
    println("Resetting incoming batch iterator")
    incoming.reset()
    outgoing.indices.foreach(outgoing(_) = null)
    outgoing(0) = new MiniBatchFileDataSetIterator(incoming.next(), batchSize / numMiniBatches)
    currBatch = 0
  }

  override def totalExamples(): Int = incoming.totalExamples()

  override def numExamples(): Int = {
    val inc = incoming.numExamples() * batchSize
    val out = outgoing(currBatch).numExamples()
    inc + out
  }

  override def next(): DataSet = {
    if (!hasNext)
      throw new IllegalStateException("Input exhausted. You should have called hasNext")
    else if (outgoing(currBatch).hasNext) {
      val n = outgoing(currBatch).next()
      n
    }
    else {
      reporters.foreach(_.report())

      if (incoming.hasNext) {
        currBatch += 1
        outgoing(currBatch) = new MiniBatchFileDataSetIterator(incoming.next, batchSize / numMiniBatches)
        val n = outgoing(currBatch).next()
        n
      } else
        throw new IllegalStateException("Input exhausted. You should have called hasNext")
    }
  }

  override def hasNext: Boolean = {
    incoming.hasNext || outgoing(currBatch).hasNext
  }
}
