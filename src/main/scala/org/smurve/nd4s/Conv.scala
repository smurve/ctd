package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

import scala.language.postfixOps

/**
  * Convolutional Layer
  *
  * @param theta        the weight matrix of size D x H x W, the first row must contain the bias in column 0
  * @param depth_input  the number of channels expected in the input vector
  * @param height_input the number of rows of the input vector
  * @param width_input  the number of columns of the input vector
  */
case class Conv(theta: INDArray, depth_input: Int, height_input: Int, width_input: Int) extends Layer {

  require(theta.rank == 3, "Expecting weight tensor of rank 3")

  val depth_theta: Int = theta.size(0)
  val height_theta: Int = theta.size(1)
  val width_theta: Int = theta.size(2)

  val depth_output: Int = depth_input * depth_theta
  val height_output: Int = height_input - (height_theta - 1) + 1 // theta has an additional row for the bias
  val width_output: Int = width_input - width_theta + 1

  /**
    * The convolution function. If you consider the given naming convention, the logic should be obvious
    * o = ouput, i = input, t = theta
    * d = depth, c = column, r = row
    *
    * @param x the input vector
    * @return the output of the convolution
    */
  def fun(x: INDArray): INDArray = {
    val output: INDArray = Nd4j.zeros(depth_output, height_output, width_output)

    /* this iterates over the entire output tensor */
    for {od <- 0 until depth_output
         or <- 0 until height_output
         oc <- 0 until width_output} {

      output(od, or, oc) = {
        /* this iterates over the weights (local receptive field) */
        val elems =
          for {tr <- 1 until height_theta
               tc <- 0 until width_theta
          } yield {
            val (id, ir, ic) = idrc(or, od, oc, tr, tc)
            x(id, ir, ic) * theta(td_od(od), tr, tc)
          }
        elems.sum + theta(td_od(od), 0, 0)
      }
    }
    output
  }

  def fwbw(x: INDArray, y_bar: INDArray): PROPAGATED = {
    val (dC_dy, grads, cost) = nextLayer.fwbw(fun(x), y_bar)
    ???
  }

  def +=(dTheta: INDArray): Unit = theta += dTheta

  /**
    * update from head and pass the tail on to subsequent layers
    *
    * @param grads : The list of gradients accumulated during training
    */
  override def update(grads: Seq[INDArray]): Unit = {
    this += grads.head
    nextLayer.update(grads.tail)
  }


  /** current depth index of the input vector */
  def id_od(od: Int): Int = od / depth_theta

  /** current depth index of the weight theta */
  def td_od(od: Int): Int = od % depth_theta

  /** current row of input vector as a function of the rows of theta and output */
  def ir_or_tr(or: Int, tr: Int): Int = or + tr - 1

  /** current column of input vector as a function of the columns of theta and output */
  def ic_oc_tc(oc: Int, tc: Int): Int = oc + tc

  /** all input indices depending on the other indices */
  def idrc(or: Int, od: Int, oc: Int, tr: Int, tc: Int): (Int, Int, Int) = (id_od(od), ir_or_tr(or, tr), ic_oc_tc(oc, tc))


}

