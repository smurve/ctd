package org.smurve.nd4s

import scala.collection.mutable

/**
  * Support for naming layer automatically and set parameters by using that name
  */
trait ParameterSupport extends Layer {

  private val parameters = mutable.Map[String, Any]()

  def integerParam(param: String) : Option[Int] = {
    parameters.get(param).map(value => Integer.parseInt(value.toString))
  }

  def stringParam(param: String) : Option[String] = {
    parameters.get(param).map(_.asInstanceOf[String])
  }

  def booleanParam(param: String) : Option[Boolean] = {
    parameters.get(param).map(_.asInstanceOf[Boolean])
  }

  override def setParams( params: (String, String, Any)* ): Unit = {
    params.filter(p=>matchesName(p._1)).foreach(param => parameters += param._2-> param._3)
    nextLayer.setParams(params: _*)
  }

  private val clazz: String = this.getClass.getSimpleName

  protected def name = s"$seqno:$clazz"

  protected def matchesName(pattern: String): Boolean = {
    pattern.split(":").toList match {
      case List("*", target) =>
        clazz == target
      case _ =>
        pattern == name
    }
  }




}
