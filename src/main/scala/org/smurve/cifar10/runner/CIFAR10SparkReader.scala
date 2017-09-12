package org.smurve.cifar10.runner

import java.io.File

import org.deeplearning4j.datasets.iterator.callbacks.FileCallback
import org.nd4j.linalg.dataset.DataSet
import org.smurve.cifar10.CIFAR10LocalContext

import scala.language.postfixOps

class CIFAR10SparkReader(context: CIFAR10LocalContext, numRecords: Int ) extends FileCallback {

  override def call[T](file: File): T = {
    val(features, labels) = context.read(file.getName, math.min(context.NUM_RECORDS_PER_FILE, numRecords))
    new DataSet(features, labels).asInstanceOf[T]
  }

}
