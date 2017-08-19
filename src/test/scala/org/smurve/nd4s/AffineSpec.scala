package org.smurve.nd4s

import org.nd4s.Implicits._
import org.scalactic.Equality
import org.scalatest.{FlatSpec, ShouldMatchers}
import org.smurve.transform._

class AffineSpec extends FlatSpec with ShouldMatchers {

  val seed = 123

  implicit val doubleEq: Equality[Double] = new Equality[Double] {
    override def areEqual(a: Double, b: Any): Boolean = a - b.asInstanceOf[Double] < 1e-7
  }

  val grid1234 = new Grid(vec(
    1, 2,
    3, 4).reshape(2, 2))

  "grid" should "support a continuous origin-centered coordinate systesm" in {

    grid1234._x(0) shouldEqual -0.5
    grid1234._y(0) shouldEqual 0.5
    grid1234._x(1) shouldEqual 0.5
    grid1234._y(1) shouldEqual -0.5

  }


  "Field values" should "match the values at the grid points" in {
    grid1234.valueAt(-.5, .5) shouldEqual 1.0
    grid1234.valueAt(-.5, -.5) shouldEqual 3.0
    grid1234.valueAt(.5, .5) shouldEqual 2.0
    grid1234.valueAt(.5, -.5) shouldEqual 4.0

  }

  "The regression plane" should "exactly fit single-component gradients" in {

    new Grid(vec(1, 1, 2, 2).reshape(2, 2)).
      valueAt(0, 0) shouldEqual 1.5

    new Grid(vec(1, 2, 1, 2).reshape(2, 2)).
      valueAt(0, 0) shouldEqual 1.5

  }


  "A Rotation of 90 degrees" should "correctly swap x and y " in {
    val grid = new Grid(vec(
      0, 0, 0,
      0, 0, 1,
      0, 0, 0)
      .reshape(3, 3))
    val pi_2 = math.Pi / 2
    val rot = Rotate(pi_2)
    val res = rot(grid)
    res.valueAt(0, 1) should equal (1.0)
    val sum = res.field.sum(0,1).getDouble(0)
    sum should equal (1.0)
  }


  private val square = new Grid(vec(
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 1, 0, 0, 0, 1, 0, 0,
    0, 0, 1, 0, 0, 0, 1, 0, 0,
    0, 0, 1, 0, 0, 0, 1, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0).reshape(9, 9) * 1.11)


  "Affine.rand" should "create a random Affine" in {
    val affine = Affine.rand(30, shear_scale_var = .2, 2, 2)
    println(square)
    println(affine(square))
  }














  private val grid87 = new Grid(vec(
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 9, 9, 9, 9, 9, 9, 9, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0).reshape(9, 9))

  "A Rotation of 45 degrees" should "turn a horizontal into a diagonal line" in {

    val rot = Rotate(math.Pi / 4 + .02)

    val transformed: Grid = rot(grid87)

    //transformed.getDouble(3,3) shouldEqual 1.0

    println(transformed)
  }

  "A Rotation of 45 degrees" should "turn a horizontal into a diagonal line (part II)" in {

    println()
    val rot = Rotate(math.Pi / 4 + .2)
    println(rot(new Grid(vec(
      0, 0, 0, 0, 0,
      0, 0, 0, 0, 0,
      1, 1, 1, 1, 1,
      0, 0, 0, 0, 0,
      0, 0, 0, 0, 0)
      .reshape(5, 5))))
  }



}
