package org.smurve.nd4s
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._

/**
  * Output layer representing the euclidean difference as cost function
  */
case class Euclidean() extends OutputLayer {

  /**
    * euclidean cost
    * @param y the output of the previous layer
    * @param y_bar the expected output
    * @return the cost at the given output y
    */
  override def cost(y: INDArray, y_bar: INDArray ): Double = {
    val diff =  y - y_bar
    val prod = diff * diff
    prod.sumT[Double] * .5
  }

  /**
    * @param y the actual output of the network that ends here
    * @param y_bar the expected output
    * @return the gradient of the euclidean cost function at the given actual output
    */
  override def grad_c(y: INDArray, y_bar: INDArray): INDArray = y - y_bar

}
