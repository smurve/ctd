package org.smurve.iter

import java.util

import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration
import org.nd4j.linalg.api.memory.enums.{AllocationPolicy, LearningPolicy}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

import scala.collection.mutable

class SimpleCIFAR10BatchIterator(dataSets: Array[DataSet], batchSize: Int ) extends DataSetIterator{

  def this(dataSet: DataSet, batchSize: Int ) = {
    this(Array(dataSet), batchSize)
  }

  val wsConfig = WorkspaceConfiguration.builder()
    .initialSize(2 * 1024L * 1024L * 1024L)
    .policyAllocation(AllocationPolicy.STRICT)
    .policyLearning(LearningPolicy.FIRST_LOOP)
    .build()

  require(dataSets.length > 0, "There should be at least one dataset")
  val N: Int = dataSets(0).getFeatures.size(0)
  for ( dataSet <- dataSets) {
    require(dataSet.getFeatures.size(0) == N, "All Feature matrices should have same size")
    require(dataSet.getFeatures.rank == 4, "Requiring training data of rank 4 in shape N x d x h x w")
    require(dataSet.getLabels.rank == 2, "Requiring training labels of rank 2 in shape N x 10")
  }

  val N_batches_per_chunk: Int = N / batchSize
  val N_chunks: Int = dataSets.length

  val batched_samples: Array[INDArray] = dataSets.map(_.getFeatures.reshape(N_batches_per_chunk, batchSize, 3, 32, 32 ))
  val batched_labels: Array[INDArray] = dataSets.map(_.getLabels.reshape(N_batches_per_chunk, batchSize, 10 ))

  var cur = 0

  override def cursor: Int = cur

  override def next(num: Int): DataSet = throw new UnsupportedOperationException("batch size is fixed upfront.")

  override def setPreProcessor(preProcessor: DataSetPreProcessor): Unit = throw new UnsupportedOperationException

  override def getPreProcessor: DataSetPreProcessor = throw new UnsupportedOperationException

  override def totalOutcomes() = 10

  override def getLabels: util.List[String] =  util.Arrays.asList(
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck")


  override def inputColumns(): Int = 3072

  override def resetSupported() = true

  override def asyncSupported = false

  override def batch(): Int = batchSize

  override def reset(): Unit = {
    //println("resetting...")
    cur = 0
  }

  override def totalExamples: Int = N

  override def numExamples: Int = N


  override def next(): DataSet = {
    //Nd4j.getWorkspaceManager.getAndActivateWorkspace(wsConfig, "Iterator")
    val cur_m = cur % N_batches_per_chunk
    val chk_m = cur / N_batches_per_chunk
    val res = new DataSet(
      batched_samples(chk_m)(cur_m, ->, ->, ->),
      batched_labels(chk_m)(cur_m, ->))
    cur += 1
    //println(s"providing new DataSet shaped ${res.getFeatures.shape().toList}")
    res
  }

  override def hasNext: Boolean = cursor < N_batches_per_chunk * N_chunks
}
