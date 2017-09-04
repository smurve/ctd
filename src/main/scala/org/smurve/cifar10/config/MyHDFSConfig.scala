package org.smurve.cifar10.config

/**
  * Configuration for HDFS environment
  */
class MyHDFSConfig extends CIFAR10Config {
  private val node = "daphnis"
  private val home = "/users/wgiersche"
  private val location = "input/cifar-10"
  override lazy val prefix = s"hdfs://$node/$home/$location"
}
