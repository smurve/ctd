package org.smurve.iter

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4s.Implicits._

class SimpleCIFAR10BatchIterator(dataSet: DataSet, batchSize: Int ) extends DataSetIterator{

  require ( dataSet.getFeatureMatrix.rank == 4, "Requiring training data of rank 4 in shape N x d x h x w" )
  require ( dataSet.getLabels.rank == 2, "Requiring training labels of rank 2 in shape N x 10" )

  val N: Int = dataSet.getFeatureMatrix.size(0)
  val N_batches: Int = N / batchSize

  val batched_samples: INDArray = dataSet.getFeatureMatrix.reshape(N_batches, batchSize, 3, 32, 32 )
  val batched_labels: INDArray = dataSet.getLabels.reshape(N_batches, batchSize, 10 )

  var cur = 0

  override def cursor: Int = cur

  override def next(num: Int): DataSet = throw new UnsupportedOperationException("batch size is fixed upfront.")

  override def setPreProcessor(preProcessor: DataSetPreProcessor): Unit = throw new UnsupportedOperationException

  override def getPreProcessor: DataSetPreProcessor = throw new UnsupportedOperationException

  override def totalOutcomes() = 10

  override def getLabels = throw new UnsupportedOperationException

  override def inputColumns(): Int = 3072

  override def resetSupported() = true

  override def asyncSupported = false

  override def batch(): Int = batchSize

  override def reset(): Unit = cur = 0

  override def totalExamples: Int = N

  override def numExamples: Int = N

  override def next(): DataSet = {
    val res = new DataSet (batched_samples (cur, ->, ->, ->), batched_labels (cur, ->) )
    cur += 1
    res
  }

  override def hasNext: Boolean = cursor < N_batches - 1
}
