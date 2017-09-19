package org.smurve.iter

import java.io.File
import java.util

import org.deeplearning4j.datasets.iterator.callbacks.FileCallback
import org.deeplearning4j.datasets.iterator.{FileSplitDataSetIterator, MultipleEpochsIterator}
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.smurve.cifar10.runner.BatchReporter

/**
  * provides iterators for lists of files
  * @param baseDirUrl the directory where the input files can be found. It's assumed that each file contains feeatures and labels
  * @param fileCallback a callback that creates a DataSet with features and labels from each file
  *
  */
class DataSetIteratorFactory ( baseDirUrl: String, fileCallback: FileCallback ) {

  /**
    *
    * @param batchReporter a reporter that does some stats after each batch
    * @param fileNames filenames to read the images from
    * @param numEpochs number of epochs, if the iterator itself is supposed to provide more than a single epoch
    * @param chunkSize the size of the chunk that is fed into the pipeline at one. Typically that's a file's content
    * @param minibatchSize
    * @return
    */
  def createIterator (batchReporter: Option[BatchReporter] = None, fileNames: Array[String], numEpochs: Int, chunkSize: Int, minibatchSize: Int ): DataSetIterator = {

    val files = fileNames.map(name => new File(baseDirUrl + "/" + name))

    val fileList: java.util.List[File] = new util.ArrayList[File](files.length)
    files.foreach(fileList.add)

    val fileSplitIterator: DataSetIterator = new FileSplitDataSetIterator(fileList, fileCallback )

    val singleEpochIterator = new IteratorPipelineProxy ( fileSplitIterator, fileNames.length, chunkSize, minibatchSize )

    batchReporter.foreach(b => singleEpochIterator.addReporter ( b ))

    if ( numEpochs > 1 )
      new MultipleEpochsIterator(numEpochs, singleEpochIterator)
    else
      singleEpochIterator
  }


}
