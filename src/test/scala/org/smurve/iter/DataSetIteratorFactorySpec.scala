package org.smurve.iter

import java.io.File

import org.deeplearning4j.datasets.iterator.callbacks.FileCallback
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.scalatest.{FlatSpec, ShouldMatchers}
import org.smurve.nd4s._
import org.nd4s.Implicits._

class DataSetIteratorFactorySpec extends FlatSpec with ShouldMatchers {

  "A DataSetIteratorFactory" should "be created from a list of files" in {

    //val reader = new CIFAR10SparkReader(new CIFAR10LocalContext("hdfs"))

    val reader = new  FileCallback {
      override def call[T](file: File): T = {
        val n = Integer.parseInt(file.getName)
        val labels = Nd4j.zeros(2, 10)
        labels(0, n) = 1.0
        labels(1, n+1) = 1.0
        new DataSet(vec(n, n+1).reshape(2,1), labels).asInstanceOf[T]
      }
    }

    val factory = new DataSetIteratorFactory("./", reader )

    val iterator = factory.createIterator(None, Array("0", "2", "4"), 2, 2, 2)


    //val iterator = factory.createIterator(Array("data_batch_1.bin", "data_batch_2.bin"), 3, 10000, 100)

    iterator.hasNext shouldBe true

    //val next: DataSet = iterator.next
    //next.getFeatures.shape() shouldEqual Array(100, 3, 32, 32)
    //next.getLabels.shape() shouldEqual Array(100, 10)

    var count = 0
    while (iterator.hasNext) {
      val n = iterator.next()
      n.getFeatures shouldEqual vec(count % 6 )
      count += 1
    }

    count shouldEqual 12
  }

}
