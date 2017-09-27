package org.smurve.cifar10.runner

import java.io.{File, FileInputStream, FileOutputStream}

import org.nd4s.Implicits._
import org.smurve.cifar10.DataContext._
import org.smurve.cifar10._

/**
  * Utility to split data into images and labels
  */
object CIFAR10DataSplitter {

  def test(args: Array[String]): Unit = {

    val res1 = new DataContext("input/cifar10").read("data_batch_1.bin", 200)
    val res2 = new DataContext("input/cifar10").readSplit("data_batch_1.bin", 200, dump = true)

    for ( i <- 0 to 100 ) {
      val px1 = res1._1(i, 0, 0, 0)
      val px2 = res2._1(i, 0, 0, 0)
      assert(px1 == px2, s"at $i: originally $px1 is now $px2")
      val l1 = res1._2(i)
      val l2 = res2._2(i)
      assert(l1 == l2, s"at $i: originally $l1 is now $l2")
    }


    println("naja...")

  }


  /**
    * split the files into images and labels
    */
  def main(args: Array[String]): Unit = {

    val fileNames = (1 to 5).map(n => s"data_batch_$n.bin").toArray :+ "test_batch.bin"

    fileNames.foreach(fileName => {
      println(s"reading $fileName")

      val orig = new Array[Byte](NUM_RECORDS_PER_FILE * BUFFER_SIZE_PER_ENTRY)
      val imgs = new Array[Byte](NUM_RECORDS_PER_FILE * IMG_SIZE)
      val lbls = new Array[Byte](NUM_RECORDS_PER_FILE)
      val fis = new FileInputStream(new File("input/cifar10/" + fileName))
      fis.read(orig)
      val fosi = new FileOutputStream(new File("input/cifar10/" + "img_" + fileName))
      val fosl = new FileOutputStream(new File("input/cifar10/" + "lbl_" + fileName))

      for (n <- 0 until NUM_RECORDS_PER_FILE) {
        val offset_orig = n * BUFFER_SIZE_PER_ENTRY
        val offset_imag = n * IMG_SIZE
        lbls(n) = orig(offset_orig)
        for (p <- 0 until IMG_SIZE) {
          imgs(offset_imag + p) = orig(offset_orig + p + 1)
        }
      }
      fosi.write(imgs)
      fosi.close()
      fosl.write(lbls)
      fosl.close()

    })
  }


}
