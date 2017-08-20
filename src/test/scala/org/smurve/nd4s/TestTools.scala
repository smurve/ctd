package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.scalactic.Equality
import org.nd4s.Implicits._
import org.scalatest.ShouldMatchers

trait TestTools extends ShouldMatchers {

  case class Precision(epsilon: Double)

  // to avoid potential interference by making a Double implicit
  implicit val PRECISION: Precision = Precision(1e-2)

  implicit val doubleEq: Equality[Double] = new Equality[Double] {
    override def areEqual(a: Double, b: Any): Boolean =
      math.abs(a - b.asInstanceOf[Double]) /
        (math.abs(a + b.asInstanceOf[Double]) + PRECISION.epsilon) < PRECISION.epsilon
  }

  /**
    * INDArray equality to a certain precision
    */
  implicit val nd4jEq: Equality[INDArray] = new Equality[INDArray] {
    override def areEqual(a: INDArray, b: Any): Boolean = {
      val r = b.asInstanceOf[INDArray].ravel
      val l = a.ravel
      require(l.shape sameElements r.shape, "Need to have the same shape to meaningfully compare.")
      (0 until l.length).forall(i => doubleEq.areEqual(l(i), r(i)))
    }
  }

  /**
    * the cost function
    *
    * @param network the network to use
    * @param y_bar   the expected output
    * @return the cost at the given input
    */
  def cost(network: Layer, y_bar: INDArray): INDArray => Double = network.fwbw(_, y_bar)._3

  /**
    * numerically compute the gradient with regards to x
    *
    * @param f the function
    * @param x the input
    * @return
    */
  def df_dx(f: (INDArray) => Double)(x: INDArray)(implicit precision: Precision): INDArray = {
    val l = x.length

    def epsvec(k: Int) = {
      val res = Nd4j.zeros(l)
      res(k) = precision.epsilon
      res.reshape(x.shape:_*)
    }

    val res = Nd4j.zeros(l)
    for (i <- 0 until x.length()) {
      res(i) = (f(x + epsvec(i)) - f(x - epsvec(i))) / 2 / precision.epsilon
    }
    res.reshape(x.shape: _*)
  }

  /**
    * Compare numerical and analytical computations of dC/dx
    *
    * @param net   the network to be tested
    * @param input the input vector
    * @param y_bar the expected output
    */
  def validateBackProp(net: Layer, input: INDArray, y_bar: INDArray): Unit = {
    val (from_backprop, _, _) = net.fwbw(input, y_bar)
    val numerical: INDArray = df_dx(cost(net, y_bar))(input)

    from_backprop shouldEqual numerical
  }

  /**
    * Compare numerical and analytical computations of dC/dTheta
    * @param net the network to be validated
    * @param theta the weight matrix to be considered
    * @param pos_theta the position of the analytical gradient in the backprop list
    * @param x input
    * @param y_bar expected outout (label)
    */
  def validateGradTheta(net: Layer, theta: INDArray, pos_theta: Int, x: INDArray, y_bar: INDArray): Unit = {

    def epsvec(k: Int) = {
      val res = Nd4j.zeros(theta.length())
      res(k) = PRECISION.epsilon
    }

    val linTheta = theta.ravel
    val numericalGrad = Nd4j.zeros(linTheta.length)

    val (_, fromBackProp, _) = net.fwbw(x, y_bar)
    for ( index <- 0 until linTheta.length ) {
      val eps = epsvec(index)
      linTheta += eps
      val (_,_,cr) = net.fwbw(x, y_bar)
      linTheta -= (epsvec(index) * 2)
      val (_,_,cl) = net.fwbw(x, y_bar)
      linTheta += epsvec(index)

      val dC_dtheta = (cr - cl) / 2 / PRECISION.epsilon
      numericalGrad(index) = dC_dtheta
    }

    fromBackProp(pos_theta) shouldEqual numericalGrad.reshape(theta.shape:_*)

  }

  /**
    * check symmetries hold for backpropagation
    */
  def checkSymmetries(net: Layer,  x: INDArray, y_bar: INDArray): Unit = {
    require(x.size(0) > 1, "Can't check symmetries with less than two input vectors")

    check_dC_dy_symmetries(net, x, y_bar)
  }


  def check_dC_dy_symmetries(net: Layer,  x: INDArray, y_bar: INDArray): Unit = {
    val (dC_dy, _, _) = net.fwbw(x, y_bar)
    val (dC_dy_0, _, _) = net.fwbw(sliceR(x, 0), sliceR(y_bar, 0))
    val (dC_dy_1, _, _) = net.fwbw(sliceR(x, 1), sliceR(y_bar, 1))
    for (i<-iArr(dC_dy_0)) {
      dC_dy_0(i:_*) shouldEqual dC_dy(i:_*)
      val i1 = i.clone()
      i1(0) = 1
      dC_dy_1(i:_*) shouldEqual dC_dy(i1:_*)
    }
  }

  /**
    * slice maintaining the rank
    * @param v the input vectors
    * @param i the index of the vector to be sliced out
    * @return a tensor of the same rank containing only the vector with given index
    */
  def sliceR ( v: INDArray, i: Int): INDArray = {
    def newShape( sub: Int*): Array[Int] = (1 +: sub).toArray
    val res = v.slice(i, 0)
    if (v.rank == 2)
      res
    else {
      val ns = newShape(res.shape(): _*)
      res.reshape(ns: _*)
    }
  }

}
