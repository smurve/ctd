package org.smurve.nd4s

import org.scalatest.{FlatSpec, ShouldMatchers}

/**
  * Just make sure that all the demos are actually able to run. Output is not validated
  */
class DemoRunnerSpec extends FlatSpec with ShouldMatchers {

  "The simple convolution demo" should "just run fine." in {
    SimpleConvolutionDemo.main(Array.empty)
  }

  "The simple optimizer demo" should "just run fine." in {
    SimpleOptimizerDemo.main(Array.empty)
  }
}
