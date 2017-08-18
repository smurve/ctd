import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4s.Implicits._
import org.smurve.nd4s._

val theta1 = vec(1,2,3,4,5,6,7,8,9).reshape(3,3)
val theta2 = vec(1,2,4,4,5,6,7,8,9).reshape(3,3)

theta1(1->,->).T

theta1.T(1->, ->)

(theta1 * theta1).sumT[Double]

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
).reshape(1, 3, 2, 4, 4)

val i2 = Nd4j.vstack(input, input)
val shape = i2.shape()

i2(0,2,1,0,1)
i2(1,2,1,0,1)

