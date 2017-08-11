package org.smurve.transform

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.smurve.nd4s._
import org.nd4s.Implicits._

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
}