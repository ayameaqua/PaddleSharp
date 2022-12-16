using Sdcb.PaddleInference.Native;
using System;
using System.Runtime.InteropServices;

namespace Sdcb.PaddleInference
{
    public class PaddleTensor : IDisposable
    {
        private IntPtr _ptr;

        public PaddleTensor(IntPtr predictorPointer)
        {
            if (predictorPointer == IntPtr.Zero)
            {
                throw new ArgumentNullException(nameof(predictorPointer));
            }
            _ptr = predictorPointer;
        }

        public string Name => PaddleNative.PD_TensorGetName(_ptr).UTF8PtrToString()!;
        public unsafe int[] Shape
        {
            get
            {
                using PaddleNative.PdIntArrayWrapper wrapper = new() { ptr = PaddleNative.PD_TensorGetShape(_ptr) };
                return wrapper.ToArray();
            }
            set
            {
                fixed (int* ptr = value)
                {
                    PaddleNative.PD_TensorReshape(_ptr, value.Length, (IntPtr)ptr);
                }
            }
        }

        public unsafe T[] GetData<T>()
        {
            TypeCode code = Type.GetTypeCode(typeof(T));
            Action<IntPtr, IntPtr> copyAction = code switch
            {
                TypeCode.Single => PaddleNative.PD_TensorCopyToCpuFloat,
                TypeCode.Int32 => PaddleNative.PD_TensorCopyToCpuInt32,
                TypeCode.Int64 => PaddleNative.PD_TensorCopyToCpuInt64,
                TypeCode.Byte => PaddleNative.PD_TensorCopyToCpuUint8,
                TypeCode.SByte => PaddleNative.PD_TensorCopyToCpuInt8,
                _ => throw new NotSupportedException($"GetData for {typeof(T).Name} is not supported.")
            };

            int[] shape = Shape;
            int size = 1;
            for (int i = 0; i < shape.Length; ++i)
            {
                size *= shape[i];
            }

            T[] result = new T[size];
            GCHandle handle = GCHandle.Alloc(result, GCHandleType.Pinned);
            copyAction(_ptr, handle.AddrOfPinnedObject());
            handle.Free();

            return result;
        }

        public unsafe void SetData(float[] data)
        {
            fixed (void* ptr = data)
            {
                PaddleNative.PD_TensorCopyFromCpuFloat(_ptr, (IntPtr)ptr);
            }
        }

        public unsafe void SetData(int[] data)
        {
            fixed (void* ptr = data)
            {
                PaddleNative.PD_TensorCopyFromCpuInt32(_ptr, (IntPtr)ptr);
            }
        }

        public unsafe void SetData(long[] data)
        {
            fixed (void* ptr = data)
            {
                PaddleNative.PD_TensorCopyFromCpuInt64(_ptr, (IntPtr)ptr);
            }
        }

        public unsafe void SetData(byte[] data)
        {
            fixed (void* ptr = data)
            {
                PaddleNative.PD_TensorCopyFromCpuUint8(_ptr, (IntPtr)ptr);
            }
        }

        public unsafe void SetData(sbyte[] data)
        {
            fixed (void* ptr = data)
            {
                PaddleNative.PD_TensorCopyFromCpuInt8(_ptr, (IntPtr)ptr);
            }
        }

        public unsafe UIntPtr[][] GetLod()
        {
            var plod = PaddleNative.PD_TensorGetLod(_ptr);
            var _pload = new IntPtr(plod.ToPointer());
            var lod = new UIntPtr[Marshal.ReadIntPtr(_pload).ToInt64()][];
            _pload = _pload + IntPtr.Size;
            for (int i = 0; i < lod.Length;i++)
            {
                var one_ptr = Marshal.ReadIntPtr(_pload);
                if(one_ptr != IntPtr.Zero)
                {
                    lod[i] = new nuint[Marshal.ReadIntPtr(one_ptr).ToInt64()];
                    one_ptr += IntPtr.Size;
                    for(int j = 0;j < lod[i].Length;j++)
                    {
                        lod[i][j] = (UIntPtr)Marshal.ReadIntPtr(one_ptr).ToPointer();
                        one_ptr += IntPtr.Size;
                    }
                }
                _pload += IntPtr.Size;
            }

            PaddleNative.PD_TwoDimArraySizeDestroy(plod);
            return lod;
        }

        public unsafe void SetLod(UIntPtr[][] array)
        {
            IntPtr lod = Marshal.AllocHGlobal((array.Length + 1) * IntPtr.Size);
            IntPtr[] data = new IntPtr[array.Length];

            int offset = 0;
            Marshal.WriteIntPtr(lod, offset, new IntPtr(array.Length));
            offset += IntPtr.Size;

            for(int i = 0;i < array.Length;i++)
            {
                data[i] = Marshal.AllocHGlobal((array[i].Length + 1) * IntPtr.Size);
                int offset2 = 0;
                Marshal.WriteIntPtr(data[i], offset2, new IntPtr(array[i].Length));
                offset2 += IntPtr.Size;
                for (int j = 0;j < array.Length;j++)
                {
                    Marshal.WriteIntPtr(data[i], offset2, (IntPtr)array[i][j].ToPointer());
                    offset2 += IntPtr.Size;
                }

                Marshal.WriteIntPtr(lod, offset, data[i]);
                offset += IntPtr.Size;
            }

            PaddleNative.PD_TensorSetLod(_ptr, lod);

            foreach(var d in data)
            {
                Marshal.FreeHGlobal(d);
            }
            Marshal.FreeHGlobal(lod);
        }

        public DataTypes DataType => (DataTypes)PaddleNative.PD_TensorGetDataType(_ptr);

        public void Dispose()
        {
            if (_ptr != IntPtr.Zero)
            {
                PaddleNative.PD_TensorDestroy(_ptr);
                _ptr = IntPtr.Zero;
            }
        }
    }
}
