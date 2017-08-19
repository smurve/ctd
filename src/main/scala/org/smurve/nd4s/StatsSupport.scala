package org.smurve.nd4s

import org.nd4j.linalg.api.ndarray.INDArray

trait StatsSupport {

  def printStats(steps: INDArray, theta: INDArray): Unit = {

    println(s"Statistics for ${getClass.getSimpleName}")
    val fmt = "%8s: min: %5.4f - max: %5.4f - avg: %5.4f - std: %5.4f"

    val avg_steps = steps.sumNumber.doubleValue / steps.length
    val std_steps = steps.stdNumber.doubleValue
    val min_steps = steps.minNumber.doubleValue
    val max_steps = steps.maxNumber.doubleValue

    val avg_theta = theta.sumNumber.doubleValue / steps.length
    val std_theta = theta.stdNumber.doubleValue
    val min_theta = theta.minNumber.doubleValue
    val max_theta = theta.maxNumber.doubleValue

    println(fmt.format("Theta", min_theta, max_theta, avg_theta, std_theta))
    println(fmt.format("Steps", min_steps, max_steps, avg_steps, std_steps))
    println()
  }

}
