package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._

import scala.collection.immutable.Seq
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.concurrent.ExecutionContext.Implicits.global

/**
  * Created by wgiersche on 26/07/17.
  */
object SimpleOptimizer {

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

    val (samples, labels) = trainingSet
    val bsize1 = trainingSet._1.size(0) / nBatches

    for (i <- 0 to n_epochs) {

      val series = 0 until nBatches
      val seq = if (parallel) series.par else series

      val (g_total, c_total): (Seq[INDArray], Double) = if (task) {

        /** Task parallelism */
        val futures: FutureResults = series.map(j => {
          val subs: INDArray = samples(j * bsize1 -> (j + 1) * bsize1, ->)
          val subl = labels(j * bsize1 -> (j + 1) * bsize1, ->)
          Future {
            val (_, grads, c) = model.fwbw(subs, subl)
            (grads, c)
          }
        })
        collectResults(futures, Nil).reduce(sum)

      } else {
        /** Data Parallelism */
        seq.map(j => {
          val subs: INDArray = samples(j * bsize1 -> (j + 1) * bsize1, ->)
          val subl = labels(j * bsize1 -> (j + 1) * bsize1, ->)
          val (_, grads, c) = model.fwbw(subs, subl)
          (grads, c)
        }).reduce(sum)
      }

      model.update(g_total.map(_ * -eta))
      if (i % reportEvery == 0)
        println(s"Cost: $c_total")
    }

    val t1 = System.currentTimeMillis()
    println(s"finished training after ${t1 - t0} ms.")

  }


}
