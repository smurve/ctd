package org.smurve.shapes

import scopt.OptionParser


class CmdLineTool() {


  val parser = new OptionParser[CmdLineArgs]("ShapeRunner") {
    head("ShapeRuner", "1.0")
    opt[Int]('b', "n_batches").valueName("Number of Batches")
      .action((x,args) => args.copy(n_batches = x))
    opt[Int]('s', "size_batch").valueName("Batch size")
      .action((x,args) => args.copy(size_batch = x))
    opt[Int]('1', "nf1").valueName("Number of features after first convolution")
      .action((x,args) => args.copy(nf1 = x))
  }
}
