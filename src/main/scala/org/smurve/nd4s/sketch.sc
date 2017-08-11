import org.nd4j.linalg.inverse.InvertMatrix
import org.nd4s.Implicits._
import org.smurve.nd4s._
import org.smurve.transform._

import scala.util.Random


val g = grid(4,3)
val value = g(0)(0)


val samples = vec(1,2,3, 4,5,6, 7,8,9, 2,3,4, 3,4,5).reshape(5,3)

samples(2,1)

val rnd = Random

samples.size(0)

samples.size(1)

val M = vec(1,2,2,4,5,6,7,8,9).reshape(3,3)

val M_ = InvertMatrix.invert(M, false)

val E = (M ** M_).ravel

val X1 = vec(0,1,1,0,0,2,1,0,3,1,1,4).reshape(4,3)
val X2 = vec(0,1,1,0,0,1,1,0,1,1,1,1).reshape(4,3)
val X3 = vec(0,1,1,0,0,1,1,0,2,1,1,2).reshape(4,3)

val X = X3

val x = X.getColumn(0)
val y = X.getColumn(1)
val z = X.getColumn(2)


val xx = (x.T ** x).getDouble(0)
val xy = (x.T ** y).getDouble(0)
val xz = (x.T ** z).getDouble(0)
val yz = (y.T ** z).getDouble(0)
val yy = (y.T ** y).getDouble(0)
val zz = (z.T ** z).getDouble(0)
val sx = x.sumNumber().doubleValue()
val sy = y.sumNumber().doubleValue()
val sz = z.sumNumber().doubleValue()
val N = X.size(0)
val b = vec(xz, yz, sz)
val C = vec(xx, xy, sx, xy, yy, sy, sx, sy, N).reshape(3,3)
val C_ = InvertMatrix.invert(C, false)
val theta = C_ ** b.T



