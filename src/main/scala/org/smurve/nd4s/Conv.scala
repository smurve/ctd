package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._

import scala.language.postfixOps

/**
  * Convolutional Layer
  * Note that we don't consider padding here
  *
  * @param theta        the weight matrix of size D x H x W, the first row must contain the bias in column 0
  * @param depth_input  the number of channels expected in the input vector
  * @param height_input the number of rows of the input vector
  * @param width_input  the number of columns of the input vector
  */
case class Conv(theta: INDArray, depth_input: Int, height_input: Int, width_input: Int,
                height_theta: Int) extends Layer {

  require(theta.rank == 3, "Expecting weight tensor of rank 3")
  require(theta.size(1)-1 == height_theta, "Theta LRFs need an additional row for the bias. Did you supply?")

  val depth_theta: Int = theta.size(0)
  val width_theta: Int = theta.size(2)

  val depth_output: Int = depth_input * depth_theta
  val height_output: Int = height_input - height_theta + 1 // theta has an additional row for the bias
  val width_output: Int = width_input - width_theta + 1

  /**
    * Multi-Indices. Keeps more-dimensional iterations concise, still readable
    */
  private val indices_output = multiIndex(0 until depth_output, 0 until height_output, 0 until width_output)
  private val indices_theta = multiIndex(0 until depth_theta, 0 until height_theta, 0 until width_theta)
  private val indices_input = multiIndex(0 until depth_input, 0 until height_input, 0 until width_input)

  /**
    * The convolution function. If you consider the given naming convention, the logic should be fairly obvious
    * o = ouput, i = input, t = theta
    * d = depth, c = column, r = row
    *
    * @param x the input vector of rank 3: D_input x H x W
    * @return the output of the convolution: rank 4! D_theta x D_input x H x W
    */
  def fun(x: INDArray): INDArray = {

    require(x.rank == 3, "Expecting rank 3: D x H x W")

    val output: INDArray = Nd4j.zeros(depth_output, height_output, width_output)

    /* this iterates over the entire output tensor using multi-index*/
    for ((od, or, oc) <- indices_output) {

      output(od, or, oc) = {
        /* this iterates over the weights (local receptive field) */
        val elems =
          for {tr <- 1 until height_theta + 1
               tc <- 0 until width_theta
          } yield {
            val (id, ir, ic) = idrc(or, od, oc, tr, tc)
            val xdrc = x(id, ir, ic)
            val tdrc = theta(td_od(od), tr, tc)
            val elem = xdrc * tdrc
            if ( (od, or, oc) == (0,10,14) ) {
              elem
            }
            elem
          }
        elems.sum + theta(td_od(od), 0, 0)
      }
    }
    output.reshape(depth_theta, depth_input, height_output, width_output)
  }

  /**
    * partial derivatives with respect to x
    */
  def dy_dx(od: Int, or: Int, oc: Int, id: Int, ir: Int, ic: Int): Double = {

    (for {tr <- 1 until height_theta + 1
          tc <- 0 until width_theta
          td = td_od(od)
    } yield
      if ((id, ir, ic) == idrc(or, od, oc, tr, tc))
        theta(td, tr, tc)
      else 0).sum
  }

  /**
    * partial derivatives with respect to theta. Concise, still readable:
    * For all input indexes, sum up the x values that have been used in fun() above
    */
  def dy_dTheta(od: Int, or: Int, oc: Int, td: Int, tr: Int, tc: Int, x: INDArray): Double =
    (for ((id, ir, ic) <- indices_input) yield
      if ((id, ir, ic) == idrc(or, od, oc, tr, tc) && td == td_od(od))
        x(id, ir, ic)
      else 0).sum

  /**
    * Compute the gradient by applying the chain rule
    *
    * @param x     the input vector
    * @param dC_dy the gradient with respect to the output (from back-prop)
    * @return the gradient
    */
  def dC_dTheta(x: INDArray, dC_dy: INDArray): INDArray = {
    val grad = Nd4j.zeros(theta.shape(): _*)
    for ((td, tr, tc) <- indices_theta) {
      grad(td, tr, tc) = (
        for ((od, or, oc) <- indices_output) yield
          dC_dy(od / depth_input, od % depth_input, or, oc) * dy_dTheta(od, or, oc, td, tr, tc, x)).sum
    }
    grad(->, 0, 0) = 1 // from the bias in position 0, 0
    grad
  }

  /**
    * Applying chain rule: dC_dx = dC_dy * dy_dx
    *
    * @param x     the input vector
    * @param dC_dy the gradient with respect to the output (from back-prop)
    * @return the cost function's derivative with regards to the input vector
    */
  def dC_dx(x: INDArray, dC_dy: INDArray): INDArray = {

    val res = Nd4j.zeros(x.shape(): _*)

    for ((id, ir, ic) <- indices_input) {
      res(id, ir, ic) = (
        for ((od, or, oc) <- indices_output)
          yield {
            if ( dy_dx(od, or, oc, id, ir, ic) != 0 )
              dC_dy(od / depth_input, od % depth_input, or, oc) * dy_dx(od, or, oc, id, ir, ic)
            else 0
          }
        ).sum
    }
    res
  }

  /**
    * forward pass and back prop in one go.
    *
    * @param x     the batch of input row vectors
    * @param y_bar the batch of expected outcome row vectors
    * @return dC/dx for the backprop, the list of all gradients, and the cost
    */
  def fwbw(x: INDArray, y_bar: INDArray): PROPAGATED = {
    val (dC_dy, grads, cost) = nextLayer.fwbw(fun(x), y_bar)
    (dC_dx(x, dC_dy), dC_dTheta(x, dC_dy) :: grads, cost)
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
  def id_od(od: Int): Int = od % depth_input

  /** current depth index of the weight theta */
  def td_od(od: Int): Int = od / depth_input

  /** current row of input vector as a function of the rows of theta and output */
  def ir_or_tr(or: Int, tr: Int): Int = or + tr - 1

  /** current column of input vector as a function of the columns of theta and output */
  def ic_oc_tc(oc: Int, tc: Int): Int = oc + tc

  /** all input indices depending on the other indices */
  def idrc(or: Int, od: Int, oc: Int, tr: Int, tc: Int): (Int, Int, Int) = (id_od(od), ir_or_tr(or, tr), ic_oc_tc(oc, tc))


}

