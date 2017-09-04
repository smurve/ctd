package org.smurve.cifar10.config

/**
  * Configuration for local file system
  */
class MyLocalConfig extends CIFAR10Config{
  override val prefix: String = "./"
}
