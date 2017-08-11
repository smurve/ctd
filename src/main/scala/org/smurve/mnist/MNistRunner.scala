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
import org.smurve.transform._

import scala.util.Random


abstract class MNistRunner ( protected val config: MNistConfig ) {

  protected val session: SparkSession
  protected val sc: SparkContext

  trait Config {
    val N_TRAINING = 30000
    val N_TEST = 1000
    val N_EPOCHS = 10
    val reportEvery = 10
    val eta: Double = 1e-5
    val nbatches = 100
    val parallelism = 4
    val task = false // task or data parallel

    val theta1: INDArray = (Nd4j.rand(seed, 785, 200) - .5) / 1000
    val theta2: INDArray = (Nd4j.rand(seed, 201, 10) - .5) / 1000
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


  def asImageString (iNDArray: INDArray) : String = {
    val arr = toArray(iNDArray)
    MNISTImage(arr.map(d=>d.toByte), 28, 28).toString
  }



  def randomTransform (
                        trans: (Double, Double),
                        angle: Integer, // in 360° units
                        shear: (Double, Double),
                        scale: (Double, Double),
                        random: Random = new Random()): Affine = {

    def rnd() = 2 * (random.nextDouble() -0.5)

    val trans_h = rnd() * trans._1
    val trans_v = rnd() * trans._2
    val rot = rnd() * angle / 360 * 2 * math.Pi
    val shear_h = rnd() * shear._1
    val shear_v = rnd() * shear._2
    val scale_h = 1 + random.nextDouble() * (scale._1 - 1)
    val scale_v = 1 + random.nextDouble() * (scale._2 - 1)

    Trans(trans_h, trans_v) °
    Rotate(rot) °
    Shear(shear_h, shear_v) °
    Scale(scale_h, scale_v)

  }

  def playWith(data: INDArray): Unit = {

    val img = new Grid(data.reshape(28,28))
    val transform = randomTransform(trans=(3,0), angle=12, shear=(.1, 0), scale=(1.1, 1))
    val td = transform(img)

    println(img)
    println(td)

    println ("ccol")

  }

  def main(args: Array[String]): Unit = {

    println("Starting...")

    // read the first sample file
    val images: MNISTImages = createImagesFromBinary("input/train").first()
    val images_vector = images.asINDarray

    // read the first label file
    val labels = createLabelsFromBinary("input/train-labels").first()
    val labels_vector = labels.asINDarray

    // read the first sample file
    val test_images: MNISTImages = createImagesFromBinary("input/test").first()
    val ti_vector = test_images.asINDarray

    // read the first label file
    val test_labels = createLabelsFromBinary("input/test-labels").first()
    val tl_vector = test_labels.asINDarray


    //playWith(images_vector.getRow(0))

    new Config {
      val nn: Layer = FCL(theta1) |:| ReLU() |:| FCL(theta2) |:| Sigmoid() |:| Euclidean()

      val probe: INDArray = images_vector.getRow(0)
      val wild_guess: INDArray = nn.ffwd(probe)
      val (d1, g1, c1): PROPAGATED = nn.fwbw(probe, labels_vector.getRow(0))

      private val generator = () => randomTransform(trans=(1,0), angle = 12, shear=(.05, .05), scale = (1.1, 1), random = new Random )

      private val optimizer = new SimpleOptimizer( generator = generator, random = new Random(seed))

      optimizer.train(model = nn, nBatches = nbatches, parallelism = parallelism, task = task,
        trainingSet = (images_vector(0->N_TRAINING, ->), labels_vector(0->N_TRAINING, ->)),
        testSet = (ti_vector(0->N_TEST, ->), tl_vector(0->N_TEST, ->)),
        n_epochs = N_EPOCHS, eta = eta, reportEvery = reportEvery)

    }

    session.stop()
  }



  protected def hdfs_save( rdd: RDD[MNISTImage], baseName: String ) : String = {
    val tmpName = baseName + "_" + UUID.randomUUID().toString
    rdd.saveAsObjectFile(tmpName)
    tmpName
  }
}
