/*
 * Copyright (c) 2019 - now, Eggroll Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.webank.eggroll.rollframe

import com.webank.eggroll.core.constant.StringConstants
import com.webank.eggroll.format.FrameDB

object TestAssets {
  val clusterManager = new ClusterManager
  def getDoubleSchema(fieldCount:Int):String = {
    val sb = new StringBuilder
    sb.append("""{
                 "fields": [""")
    (0 until fieldCount).foreach{i =>
      if(i > 0) {
        sb.append(",")
      }
      sb.append(s"""{"name":"double$i", "type": {"name" : "floatingpoint","precision" : "DOUBLE"}}""")
    }
    sb.append("]}")
    sb.toString()
  }

  def loadCache(path: String,storeType:String): Unit = {
    val inputDB = storeType match {
      case StringConstants.HDFS => FrameDB.hdfs(path)
      case _ => FrameDB.file(path)
    }

    val outputDB = FrameDB.cache(path)
    outputDB.writeAll(inputDB.readAll())
    outputDB.close()
    inputDB.close()
  }
}
