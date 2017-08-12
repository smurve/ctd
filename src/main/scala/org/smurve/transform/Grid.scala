package org.smurve.transform

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.inverse.InvertMatrix
import org.nd4s.Implicits._
import org.smurve.nd4s._

/**
  * A Grid represents a function of 2 parameters that is represented by values at certain grid points
  * To support origin-based transformations such as rotations, we maintain a continuous coordinate system
  * with a center right in the middle of the grid. The origin may not have a discrete representation in the grid.
  * Example: a 3x4 grid spanning the orgigin-based coordinates x = [-1.5, -0.5, 0.5, 1.5], y = [-1, 0, 1]
  *
  *                 |
  *           *   * | *   *
  *                 |
  * --------- * - * + * - * -----------
  *                 |
  *           *   * | *   *
  *                 |
  *
  */
class Grid(val field: INDArray, val useRegression: Boolean = false) {

  val height: Int = field.size(0)
  val width: Int = field.size(1)
  val min: Double = field.minT[Double]
  val max: Double = field.maxT[Double]


  /** the top-left origin-centered coordinates */
  val left_oc: Double = -(width - 1) / 2.0
  val top_oc: Double = (height - 1) / 2.0

  case class Sector(top: Int, left: Int, bottom: Int, right: Int) {
    require(top >= 0)
    require(left >= 0)
    require(bottom < height, "bottom must be smaller than height")
    require(right < width)
  }

  /**
    * mapping from and to the origin-centered coordinate system
    */
  /** the closest left colomn index */
  def _c(x: Double): Int = (x - left_oc).toInt
  /** the closest upper row index */
  def _r(y: Double): Int = (top_oc - y ).toInt
  /** origin_centered x value for column */
  def _x(c: Int): Double = c + left_oc
  /** origin_centered y value for row */
  def _y(r: Int): Double = top_oc - r

  /**
    * @param x ab-origin x coordinate
    * @param y ab-origin y coordinate
    * @return Array of 4 closest neighbours of the given point in index
    */
  def neighbours(x: Double, y: Double) = Sector(top = _r(y), left = _c(x), bottom = _r(y)+1, right = _c(x)+1)


  /**
    * determines the value of the function represented by the grid.
    * On the points, returns the actual field value - between the points, use interpolation or regression
    * Due to switching regression params and immediate values, this function is not continuous
    *
    * @param x origin-centered x-coordinate
    * @param y origin-centered y-coordinate
    * @return the non-continuous value of the function represented by the grid
    */
  def valueAt(x: Double, y: Double): Double = {

    /* beyond borders */
    val res = if (x < left_oc || x > -left_oc || y < -top_oc || y > top_oc)
      0.0

    /* right on the points */
    else if (x - left_oc == _c(x) && top_oc - y == _r(y)) {
      field(_r(y), _c(x))
    }

    /* interpolation on y */
    else if (x - left_oc == _c(x)) {
      val r1 = _r(y)
      val z1 = field(r1, _c(x))
      val r2 = r1 + 1
      val z2 = field(r2, _c(x))
      z1 + (z2 - z1) / -1  /* / (y2 - y1) */ * (y - _y(r1))

      /* interpolation on x */
    } else if (top_oc - y == _r(y)) {
      val c1 = _c(x)
      val z1 = field(c1, _r(y))
      val c2 = c1 + 1
      val z2 = field(c2, _r(y))
      z1 + (z2 - z1) /* / (x2 - x1) */ * (x - _x(c1))
    }

    /* 2D regression */
    else {
      val sector = neighbours(x, y)
      if ( useRegression ) {
        val (a, b, c) = linReg(sector)
        a * x + b * y + c
      } else { // use average from neighbours
        val tl = valueAt(_x(sector.left), _y(sector.top))
        val bl = valueAt(_x(sector.left), _y(sector.bottom))
        val tr = valueAt(_x(sector.right), _y(sector.top))
        val br = valueAt(_x(sector.right), _y(sector.bottom))
        (tl + bl + tr + br)/4
      }
    }

    math.max(res, 0)
  }

  /**
    * Calculate the regression plane in the given sector analytically.
    * Computationally expensive. Use only if performance is none of your concern.
    *
    * @param sector the target sector
    * @return the parameters a, b, and c of the plane, such that z = a*x + b*y + c
    */
  def linReg(sector: Sector): (Double, Double, Double) = {

    val (sl, st) = (_x(sector.left), _y(sector.top))

    val c = vec(sl, sl, sl + 1, sl + 1)
    val r = vec(st, st - 1, st, st - 1)
    val z = vec(
      field(sector.top, sector.left),
      field(sector.top + 1, sector.left),
      field(sector.top, sector.left + 1),
      field(sector.top + 1, sector.left + 1))

    val N = 4
    val xx = (c ** c.T).getDouble(0)
    val xy = (c ** r.T).getDouble(0)
    val xz = (c ** z.T).getDouble(0)
    val yy = (r ** r.T).getDouble(0)
    val yz = (r ** z.T).getDouble(0)
    val sx = c.sumNumber().doubleValue()
    val sy = r.sumNumber().doubleValue()
    val sz = z.sumNumber().doubleValue()
    val b = vec(xz, yz, sz)
    val C = vec(xx, xy, sx, xy, yy, sy, sx, sy, N).reshape(3, 3)
    val C_ = InvertMatrix.invert(C, false)
    val theta = C_ ** b.T
    (theta(0), theta(1), theta(2))
  }

  def scaleToByte(x: Double): Byte = {
    (255 * (min + (x - min)/(max - min))).toByte
  }

  override def toString: String = {

  ( 0 until height ).map (i => {
    val arr = toArray(field(i, ->))
    val row = arr.map(scaleToByte)
    rowAsString(row)
  }).mkString("\n")}


  private def rowAsString ( bytes: Array[Byte]) : String = {
    bytes.map(b=>{
      val n  = b & 0xFF
      val c = if (n == 0) 0 else n / 32 + 1
      c match {
        case 0 => "  "
        case 1 => "' "
        case 2 => "''"
        case 3 => "::"
        case 4 => ";;"
        case 5 => "cc"
        case 6 => "OO"
        case 7 => "00"
        case 8 => "@@"
      }
    }).mkString("")
  }
}