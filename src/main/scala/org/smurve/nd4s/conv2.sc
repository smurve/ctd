import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.smurve.nd4s._
import org.nd4s.Implicits._

val theta1 = Nd4j.rand(Array(10,3,3))-.5
val theta2 = Nd4j.rand(Array(10, 3, 3)) -0.5

val conv1 = Conv(theta1, 1, 28, 28)

// INDArrays are row vectors
val v = vec(1,2,3, 4)
val v2 = v ** v.T

val v1 = vec(1,2,3,4)

v1 == v
v1 == v.T
v1 == v.reshape(2,2)

def f(x: Double) = x * x * x

val epsilon = 1e-3

(f(3+epsilon) - f(3-epsilon)) / 2 / epsilon

val theta = vec (
  1,2,2,3,
  1,-1,1,1,
  2,2,1,1).reshape(4,3)

val x=vec(3,2,1,3,2,1).reshape(3,2)

def f(x: INDArray ) = theta ** x

f(x)(0,0)


theta.length()