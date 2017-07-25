package org.smurve.mnist

import java.util.UUID

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

abstract class MNistRunner ( protected val config: MNistConfig ) {

  protected val session: SparkSession
  protected val sc: SparkContext
  protected def hdfs ( name: String ): String = config.prefix + name


  protected def createImagesromBinary(name: String): RDD[MNISTImages] = {

    val rawImages = sc.binaryFiles(hdfs(name))

    /*
    val files = rawImages.collect()
    val stream = files(0)._2
    val bytes = stream.toArray()
    */

    rawImages.map(p=>{
      val stream = p._2
      new MNISTImages(stream.toArray())
    })

    //new MNISTImages(bytes)
  }

  def main(args: Array[String]): Unit = {

    println("Starting...")

    val imgs: MNISTImages = createImagesromBinary("input/train").first()
    val rdd: RDD[MNISTImage] = sc.parallelize(imgs.sequence)
    //val tmpName = hdfs_save(rdd, hdfs("/temp/images"))
    val img = rdd.first()

    println(img)

    println("Done.")
    session.stop()
  }

  protected def hdfs_save( rdd: RDD[MNISTImage], baseName: String ) : String = {
    val tmpName = baseName + "_" + UUID.randomUUID().toString
    rdd.saveAsObjectFile(tmpName)
    tmpName
  }
}
