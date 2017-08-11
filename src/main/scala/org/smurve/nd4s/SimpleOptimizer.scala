package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._
import org.smurve.transform.Affine

import scala.collection.immutable.Seq
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.Random

/**
  * Created by wgiersche on 26/07/17.
  */
class SimpleOptimizer(val generator: () => Affine,
                      random: Random) {


  type FutureResults = Seq[Future[(List[INDArray], Double)]]
  type Results = List[(List[INDArray], Double)]

  def sum(a: (Seq[INDArray], Double), b: (Seq[INDArray], Double)): (Seq[INDArray], Double) = {
    val avgNabla = (a._1 zip b._1) map { case (l, r) => l + r }
    val avgC = a._2 + b._2
    (avgNabla, avgC)
  }

  def collectResults(todo: FutureResults, done: Results): Results = {
    if (todo.isEmpty)
      done
    else {
      val res = Await.result(todo.head, Duration.Inf)
      res :: collectResults(todo.tail, res :: done)
    }
  }

  def train(model: Layer, nBatches: Int, parallel: Boolean, task: Boolean,
            trainingSet: (INDArray, INDArray), n_epochs: Int, eta: Double, reportEvery: Int): Unit = {

    println("Started training")
    val t0 = System.currentTimeMillis()

    val batchSize = trainingSet._1.size(0) / nBatches

    for (epoch <- 0 to n_epochs) {

      val (samples, labels) = shuffle(trainingSet)


      val series = 0 until nBatches
      val seq = if (parallel) series.par else series

      val (g_total, c_total): (Seq[INDArray], Double) = if (task) {

        /** Task parallelism */
        val futures: FutureResults = series.map(j => {
          val subs: INDArray = samples(j * batchSize -> (j + 1) * batchSize, ->)
          val subl = labels(j * batchSize -> (j + 1) * batchSize, ->)
          Future {
            val (_, grads, c) = model.fwbw(subs, subl)
            (grads, c)
          }
        })
        collectResults(futures, Nil).reduce(sum)

      } else {
        /** Data Parallelism */
        seq.map(j => {
          val subs: INDArray = samples(j * batchSize -> (j + 1) * batchSize, ->)
          val subl = labels(j * batchSize -> (j + 1) * batchSize, ->)
          val (_, grads, c) = model.fwbw(subs, subl)
          (grads, c)
        }).reduce(sum)
      }


      model.update(g_total.map(_ * -eta))
      if (epoch % reportEvery == 0)
        println(s"Cost: $c_total")
    }

    val t1 = System.currentTimeMillis()
    println(s"finished training after ${t1 - t0} ms.")

  }

  def train(model: Layer, nBatches: Int, parallelism: Integer, task: Boolean,
            trainingSet: (INDArray, INDArray), testSet: (INDArray, INDArray),
            n_epochs: Int, eta: Double, reportEvery: Int): Unit = {

    assert(parallelism >= 1, "Parallelism can't be less than 1.")

    println("Started training")
    val t0 = System.currentTimeMillis()

    val batchSize = trainingSet._1.size(0) / nBatches
    val blockSize = batchSize / parallelism
    val nBlocks = batchSize / blockSize

    for (epoch <- 1 to n_epochs) {

      println(s"Starting epoch $epoch")

      val (samples, labels) = shuffle(trainingSet, random = random, transform = generator.apply())

      for (batchNo <- 0 until nBatches) {

        val offset = batchNo * batchSize
        val blocks = (0 until nBlocks).par

        val (g_total, c_total): (Seq[INDArray], Double) =
          blocks.map(block => {
            val fromIndex = offset + block * blockSize
            val toIndex = offset + (block + 1) * blockSize
            val subs: INDArray = samples(fromIndex -> toIndex, ->)
            val subl = labels(fromIndex -> toIndex, ->)
            val (_, grads, c) = model.fwbw(subs, subl)
            (grads, c)
          }).reduce(sum)

        model.update(g_total.map(_ * -eta))
        if (batchNo % reportEvery == 0)
          println(s"Cost: $c_total")
      }

      val N_TEST = testSet._2.size(0)
      val res = model.ffwd(testSet._1)

      val success = ( 0 until N_TEST).map( i=>{
        val pred = res(i,->)
        val label = testSet._2(i, ->)
        if ( equiv( pred, label) ) 1.0 else 0.0
      }).sum / N_TEST

      println( s"Success rate: ${(success*1000).toInt/10.0}")
    }

    val t1 = System.currentTimeMillis()
    println(s"finished training after ${t1 - t0} ms.")

  }


}
