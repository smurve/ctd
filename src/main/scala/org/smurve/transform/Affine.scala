package org.smurve.transform

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.smurve.nd4s._
import org.nd4s.Implicits._

import scala.util.Random

/**
  * 2-DIM Affine transformation for image processing
  */
trait Affine  {

  private val me = this

  /**
    * the inverse transformation to be applied to the underlying coordinate system.
    * @param x the original x
    * @return the new coordinate
    */
  def invMap(x: INDArray ): INDArray

  /**
    * @param other the right-hand side of the composition expression
    * @return the Affine representing the composition expression
    */
  def ° ( other: Affine ): Affine = new Affine {
    override def invMap(x:INDArray): INDArray = other.invMap(me.invMap(x))
  }

  /**
    * for those who don't like operator overloading, or particularly the operator "°"
    * @param other the right-hand side of the composition expression
    * @return the Affine representing the composition expression
    */
  def compose ( other: Affine): Affine = °( other)

  /**
    * Apply the transformation on the grid.
    *
    * @param grid the grid
    * @return a new grid after applying the affine transformation
    */
  def apply(grid: Grid): Grid = {
    require(grid.field.rank == 2, "Field should be 2-dimensional")

    val newField = Nd4j.create(grid.field.shape: _*)

    for ( row <- 0 until grid.height )
      for (col <- 0 until grid.width) {
        val xy = vec(grid._x(col),grid._y(row))
        val m = invMap( xy )
        val value = grid.valueAt(m(0), m(1))
        newField(row, col) = value
      }

    new Grid ( newField )
  }


}

object Affine {

  val identity = new Affine {
    override def invMap(x: INDArray): INDArray = x
  }

  /**
    * create an Affine transformation from the parameters of the *inverse* transformation
    *
    * @param t00 x to x of the linear matrix
    * @param t01 x to y of the linear matrix
    * @param t10 y to x of the linear matrix
    * @param t11 y to x of the linear matrix
    * @param x translate x
    * @param y translate y
    * @return
    */
  def fromParams ( t00:Double, t01:Double, t10:Double, t11:Double, x:Double, y:Double ): Affine = new Affine {

    private val b = vec(x,y)
    private val _M = vec(t00, t01, t10, t11).reshape(2,2)

    override def invMap(x: INDArray): INDArray = {
      (_M ** x.T + b.T).T
    }
  }

  def rand(maxAngle: Int = 0, shear_scale_var: Double = 0,
           max_trans_x: Double = 0, max_trans_y: Double = 0,
           random: Random = new Random): Affine = {

    def rnd() = 2 * (random.nextDouble() - 0.5)

    val sh = shear_scale_var * rnd()
    val angle = maxAngle / 360.0 * 2 * math.Pi * rnd()
    val c = math.cos(angle) * (1 + rnd() * shear_scale_var)
    val s = math.sin(angle) * (1 + rnd() * shear_scale_var)
    val x = max_trans_x * rnd()
    val y = max_trans_y * rnd()
    fromParams(c + sh*s, s + sh*c, -s+sh*c, c+sh*s, x, y)
  }


}