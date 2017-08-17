import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.smurve.nd4s._
import org.nd4s.Implicits._

val theta1 = Nd4j.rand(Array(10,3,3))-.5
val theta2 = Nd4j.rand(Array(10, 3, 3)) -0.5

val conv1 = Conv(theta1, 1, 28, 28)

val input: INDArray = vec(
  -2, -2, -2, -2,
  -2, -2, -2, -2,
  -2, -2, -2, -2,
  -2, -2, -2, -2,

  2, 2, 2, 2,
  2, 2, 2, 2,
  2, 2, 2, 2,
  2, 2, 2, 2,

  4, 4, 4, 4,
  4, 4, 4, 4,
  4, 4, 4, 4,
  4, 4, 4, 4,

  4, 4, 4, 4,
  4, 4, 4, 4,
  4, 4, 4, 4,
  4, 4, 4, 4,

  1, 2, 3, 4,
  2, 3, 4, 5,
  3, 4, 5, 6,
  4, 5, 6, 7,

  4, 5, 6, 7,
  3, 4, 5, 6,
  2, 3, 4, 5,
  1, 2, 3, 4
).reshape(3, 2, 4, 4)

val(od, or, oc) = (2, 0, 0)
val slice = input(od, 0 -> 2, 2 * or-> 2* (or+1), oc*2 -> 2* (oc+1))
val max1 = slice.max(0,1,2)






