import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.nd4s.Implicits._
import org.smurve.nd4s._

val theta1 = vec(1,2,3,4,5,6,7,8,9).reshape(3,3)
val theta2 = vec(1,2,4,4,5,6,7,8,9).reshape(3,3)

theta1(1->,->).T

theta1.T(1->, ->)

(theta1 * theta1).sumT[Double]
