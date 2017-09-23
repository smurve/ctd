package org.smurve.iter

import org.scalatest.{FlatSpec, ShouldMatchers}


class FileBufferIteratorTest extends FlatSpec with ShouldMatchers {

  "A FileBufferIterator" should "iterate, obviously..." in {

    val files = (1 to 2).map(n=>s"./input/cifar10/data_batch_$n.bin").toArray
    val it = new FileBufferIterator(files, 2500 * 3073)

    for ( _ <- 1 to 2 ) {
      it.hasNext shouldBe true

      (1 to 4).foreach(_ => {
        val next = it.next
        next(0).toInt should be < 10
      })

      it.hasNext shouldBe true

      it.next

      it.reset()

    }
  }

}
