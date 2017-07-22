package org.smurve.mnist

/**
  * Created by wgiersche on 22/07/17.
  */
class HDFSConfig extends MNistConfig {
  private val node = "daphnis"
  private val home = "/users/wgiersche"
  override lazy val prefix = s"hdfs://$node/$home/"
}
