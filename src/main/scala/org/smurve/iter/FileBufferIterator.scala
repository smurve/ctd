package org.smurve.iter

import java.io.{FileInputStream, InputStream}

/**
  * attempts to read chunks of fixed size from a set of files. Iterates through the list until all data is read
  *
  * WARNING: INTENTIONALLY NOT THREAD-SAFE!!!
  * Initially allocates an array of bytes and uses the same array again and again
  * Will fail if any of the read attempts returns less than the chunk size of bytes
  */
class FileBufferIterator(files: Array[String], chunkSize: Int ) {

  require ( files.length > 0 )

  private val buffer = new Array[Byte](chunkSize)

  private var curFileIndex = 0
  private var curStream: InputStream = new FileInputStream(files(0))
  private var cur = 0

  def cursor: Int = cur

  def hasNext: Boolean = curFileIndex < files.length - 1 || curStream.available() > 0

  def next: Array[Byte] = {
    if ( curStream.available() > 0 ) {
      val nBytes = curStream.read(buffer)
      if ( nBytes < chunkSize) {
        throw new IllegalStateException(s"Failed to load $chunkSize bytes. Got only $nBytes")
      }
      cur += 1
      buffer
    }
    else if (curFileIndex < files.length - 1) {
      curStream.close()
      curFileIndex += 1
      curStream = new FileInputStream(files(curFileIndex))
      next
    } else {
      curStream.close()
      throw new NoSuchElementException("Don't have any more data available")
    }
  }

  def reset(): Unit = {
    cur = 0
    curFileIndex = 0
    curStream = new FileInputStream(files(0))
  }
}
