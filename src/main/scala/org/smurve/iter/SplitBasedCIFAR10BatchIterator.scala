package org.smurve.iter

import java.util

import org.deeplearning4j.datasets.iterator.{FloatsDataSetIterator, INDArrayDataSetIterator}
import org.nd4j.linalg.api.buffer.FloatBuffer
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration
import org.nd4j.linalg.api.memory.enums.{AllocationPolicy, LearningPolicy}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.{DataSet, ExistingMiniBatchDataSetIterator, MiniBatchFileDataSetIterator}
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.primitives
import org.nd4s.Implicits._
import org.smurve.cifar10.DataContext._

import scala.collection.immutable


/**
  * Iterator to produce DataSets (mini-batches) from any number of files. We assume the inner structure of the file
  * as described for the binary format in https://www.cs.toronto.edu/~kriz/cifar.html, only that labels and samples
  * are split into two files each, to simplify the ingestion
  *
  * WARNING: INTENTIONALLY NOT THREAD-SAFE!!!
  * pre-allocates byte arrays and NDArrays to minimize memory footprint and GC overhead
  *
  * @param files     the list of absolut file names
  * @param batchSize the number of images to be produced with each call to next*
  */
class SplitBasedCIFAR10BatchIterator(basePath: String, files: Array[String], batchSize: Int) extends DataSetIterator {

  var tin: Long = 0
  var tout: Long = 0

  var cur = 0
  require(files.length > 0, "There should be at least one file to read from")
  val imageBufferIterator = new FileBufferIterator(files.map(s"$basePath/img_" + _), batchSize * IMG_SIZE)
  val labelBufferIterator = new FileBufferIterator(files.map(s"$basePath/lbl_" + _), batchSize)

  private val iteratorConfig = WorkspaceConfiguration.builder
    .policyAllocation(AllocationPolicy.STRICT)
    .policyLearning(LearningPolicy.FIRST_LOOP)
    .build // <-- this option makes workspace learning after first loop


  // pre-allocated NDArrays
  private val labels = Nd4j.zeros(batchSize * 10).reshape(batchSize, 10)
  private val images = Nd4j.zeros(batchSize * IMG_SIZE).reshape(batchSize, NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

  override def cursor: Int = imageBufferIterator.cursor

  override def next(num: Int): DataSet = throw new UnsupportedOperationException("batch size is fixed upfront.")

  override def setPreProcessor(preProcessor: DataSetPreProcessor): Unit = throw new UnsupportedOperationException

  override def getPreProcessor: DataSetPreProcessor = throw new UnsupportedOperationException

  override def totalOutcomes() = 10

  override def getLabels: util.List[String] = util.Arrays.asList(
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
    imageBufferIterator.reset()
    labelBufferIterator.reset()
    cur = 0
  }

  override def totalExamples: Int = throw new UnsupportedOperationException("Can't estimate")

  override def numExamples: Int = throw new UnsupportedOperationException("Can't estimate")


  override def next: DataSet = {

    val ws = Nd4j.getWorkspaceManager.getAndActivateWorkspace(iteratorConfig, "Iterator")

    val nextImageBuffer = imageBufferIterator.next
    val nextLabelBuffer: Array[Byte] = labelBufferIterator.next
    val labels = Nd4j.zeros(batchSize * 10).reshape(batchSize, 10)

    for (n <- 0 until batchSize) {
      labels(n, nextLabelBuffer(n).toInt) = 1f
    }

    val arr = nextImageBuffer.map(b => (b & 0xFF).toFloat) // / 256f - 0.5 )
    //images.setData(new FloatBuffer(arr))
    val images = Nd4j.create(arr).reshape(batchSize, NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

    // save mem and time by doing the centering in-place on the NDArray
    images.divi(256f).subi(.5f)

    new DataSet(images, labels)

  }

  override def hasNext: Boolean = imageBufferIterator.hasNext
}
