package org.smurve.iter

import org.scalatest.{FlatSpec, ShouldMatchers}
import org.smurve.cifar10._

class FileBasedCIFAR10BatchIteratorSpec extends FlatSpec with ShouldMatchers {

  "An Iterator" should "produce images and labels from the list of files" in {

    val files = (1 to 2).map(n=>s"data_batch_$n.bin").toArray

    val it = new SplitBasedCIFAR10BatchIterator("input/cifar10/", files, 500 )

    while ( it.hasNext ) {
      val batch = it.next
      dumpAsImages(batch, 5)
    }
  }
}
