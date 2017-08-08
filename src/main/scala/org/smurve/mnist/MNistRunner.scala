package org.smurve.mnist

import java.util.UUID

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.smurve.mnist.config.MNistConfig
import org.smurve.nd4s.SimpleOptimizerDemo.{matches, seed}
import org.smurve.nd4s._


abstract class MNistRunner ( protected val config: MNistConfig ) {

  protected val session: SparkSession
  protected val sc: SparkContext

  trait Setup {
    val N_EPOCHS = 1
    val reportEvery = 10
    val eta: Double = 3e-5
    val nbatches = 1000
    val parallel = false
    val task = false // task or data parallel

    val theta1: INDArray = (Nd4j.rand(seed, 785, 100) - .5) / 1000
    val theta2: INDArray = (Nd4j.rand(seed, 101, 10) - .5) / 1000
  }

  /**
    * resolve in local fs or hdfs, depending on config
    * @param name short name of the file
    * @return
    */
  protected def resolve(name: String ): String = config.prefix + name


  /**
    * @param name name of file or directory to read from
    * @return an RDD of MNISTImages wrappers
    */
  protected def createImagesFromBinary(name: String): RDD[MNISTImages] = {

    val rawImages = sc.binaryFiles(resolve(name))

    rawImages.map(p=>{
      val stream = p._2
      new MNISTImages(stream.toArray())
    })
  }

  /**
    * @param name name of file or directory to read from
    * @return an RDD of MNISTImages wrappers
    */
  protected def createLabelsFromBinary(name: String): RDD[MNISTLabels] = {

    val rawLabels = sc.binaryFiles(resolve(name))

    rawLabels.map(p=>{
      val stream = p._2
      new MNISTLabels(stream.toArray())
    })
  }


  def main(args: Array[String]): Unit = {

    println("Starting...")

    // read the first sample file
    val images: MNISTImages = createImagesFromBinary("input/train").first()
    val images_vector = images.asINDarray

    // read the first label file
    val labels = createLabelsFromBinary("input/train-labels").first()
    val labels_vector = labels.asINDarray

    shuffle((images_vector, labels_vector))

    // verify that the loaded and shuffled data is correct
    ( 0 to 10 ).foreach (i => {
      val row: INDArray = images_vector.getRow(i)
      println(asString(row))
      println("  ======================================>  " + labels.asINDarray.getRow(i))
    })

    new Setup {
      val nn: Layer = FCL(theta1) |:| ReLU() |:| FCL(theta2) |:| Sigmoid() |:| Euclidean()

      val probe: INDArray = images_vector.getRow(0)
      val wild_guess: INDArray = nn.ffwd(probe)
      val (d1, g1, c1): PROPAGATED = nn.fwbw(probe, labels_vector.getRow(0))

      SimpleOptimizer.train(model = nn, nBatches = nbatches, parallelism = 1, task = task,
        trainingSet = (images_vector, labels_vector), n_epochs = N_EPOCHS, eta = eta, reportEvery = reportEvery)

      val testSize = 100
      val (samples, labels) = (images_vector(0 -> testSize, ->), labels_vector(0 -> testSize, ->))
      val educated_guess: INDArray = nn.ffwd(samples)

      for (i <- 0 until testSize) {
        if (!matches(educated_guess(i, ->), labels(i, ->))) {
          println(s"For ${asString(samples(i, ->))} got ${educated_guess(i, ->)}, but would expect ${labels(i, ->)}")
        }
      }

      val success_rate: Double = (0 until testSize).
        map(i => if (matches(educated_guess(i, ->), labels(i, ->))) 1 else 0).sum * 100.0 / testSize

      println(s"Success rate: $success_rate")

    }



    session.stop()
  }



  protected def hdfs_save( rdd: RDD[MNISTImage], baseName: String ) : String = {
    val tmpName = baseName + "_" + UUID.randomUUID().toString
    rdd.saveAsObjectFile(tmpName)
    tmpName
  }
}
