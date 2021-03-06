/*
 * Copyright 2019 The Eggroll Authors. All Rights Reserved.
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

// automatically generated by the FlatBuffers compiler, do not modify

package com.webank.ai.eggroll.core.io;

import com.google.flatbuffers.FlatBufferBuilder;
import com.google.flatbuffers.Table;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

@SuppressWarnings("unused")
public final class Operand extends Table {
    public static Operand getRootAsOperand(ByteBuffer _bb) {
        return getRootAsOperand(_bb, new Operand());
    }

    public static Operand getRootAsOperand(ByteBuffer _bb, Operand obj) {
        _bb.order(ByteOrder.LITTLE_ENDIAN);
        return (obj.__assign(_bb.getInt(_bb.position()) + _bb.position(), _bb));
    }

    public static int createOperand(FlatBufferBuilder builder,
                                    int keyOffset,
                                    int valueOffset) {
        builder.startObject(2);
        Operand.addValue(builder, valueOffset);
        Operand.addKey(builder, keyOffset);
        return Operand.endOperand(builder);
    }

    public static void startOperand(FlatBufferBuilder builder) {
        builder.startObject(2);
    }

    public static void addKey(FlatBufferBuilder builder, int keyOffset) {
        builder.addOffset(0, keyOffset, 0);
    }

    public static int createKeyVector(FlatBufferBuilder builder, byte[] data) {
        builder.startVector(1, data.length, 1);
        for (int i = data.length - 1; i >= 0; i--) builder.addByte(data[i]);
        return builder.endVector();
    }

    public static void startKeyVector(FlatBufferBuilder builder, int numElems) {
        builder.startVector(1, numElems, 1);
    }

    public static void addValue(FlatBufferBuilder builder, int valueOffset) {
        builder.addOffset(1, valueOffset, 0);
    }

    public static int createValueVector(FlatBufferBuilder builder, byte[] data) {
        builder.startVector(1, data.length, 1);
        for (int i = data.length - 1; i >= 0; i--) builder.addByte(data[i]);
        return builder.endVector();
    }

    public static void startValueVector(FlatBufferBuilder builder, int numElems) {
        builder.startVector(1, numElems, 1);
    }

    public static int endOperand(FlatBufferBuilder builder) {
        int o = builder.endObject();
        return o;
    }

    public void __init(int _i, ByteBuffer _bb) {
        bb_pos = _i;
        bb = _bb;
    }

    public Operand __assign(int _i, ByteBuffer _bb) {
        __init(_i, _bb);
        return this;
    }

    public int key(int j) {
        int o = __offset(4);
        return o != 0 ? bb.get(__vector(o) + j * 1) & 0xFF : 0;
    }

    public int keyLength() {
        int o = __offset(4);
        return o != 0 ? __vector_len(o) : 0;
    }

    public ByteBuffer keyAsByteBuffer() {
        return __vector_as_bytebuffer(4, 1);
    }

    public ByteBuffer keyInByteBuffer(ByteBuffer _bb) {
        return __vector_in_bytebuffer(_bb, 4, 1);
    }

    public int value(int j) {
        int o = __offset(6);
        return o != 0 ? bb.get(__vector(o) + j * 1) & 0xFF : 0;
    }

    public int valueLength() {
        int o = __offset(6);
        return o != 0 ? __vector_len(o) : 0;
    }

    public ByteBuffer valueAsByteBuffer() {
        return __vector_as_bytebuffer(6, 1);
    }

    public ByteBuffer valueInByteBuffer(ByteBuffer _bb) {
        return __vector_in_bytebuffer(_bb, 6, 1);
    }
}

