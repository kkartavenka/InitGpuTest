using Accord.Statistics;
using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.ScanReduceOperations;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;
using System;
using System.Diagnostics;
using System.Linq;

namespace InitGpuTest
{
    class Program
    {
        static float[] Arr;

        static void Main() {

            int n = 10000000;
            Arr = new float [n];
            int r = 10;

            for (int i = 0; i < n; i++) Arr[i] = (float)new Random().NextDouble();

            #region CPU test
            Stopwatch sw = new();
            
            sw.Start();
            for (int i = 0; i < r; i++) {
                float sdCpu = (float)Arr.Select(m => (double)m).ToArray().StandardDeviation();
            }
            sw.Stop();

            Console.WriteLine($"CPU: {sw.ElapsedMilliseconds}");
            #endregion

            sw.Reset();

            #region GPU test
            using var context = new Context();

            context.EnableAlgorithms();
            sw.Start();

            using var accelerator = new CudaAccelerator(context);

            for (int i = 0; i < r; i++) {
                using var sourceBuffer = accelerator.Allocate(Arr);
                using var meanValueBuffer = accelerator.Allocate<float>(1);
                using var sumPoweredBuffer = accelerator.Allocate<float>(Arr.Length);
                using var sumValueBuffer = accelerator.Allocate<float>(1);

                accelerator.Reduce<float, AddFloat>(accelerator.DefaultStream, sourceBuffer.View, meanValueBuffer.View);
                
                var loadedKernel = accelerator.LoadAutoGroupedStreamKernel((Index1 i, ArrayView<float> data, long value) => data[i] = data[i] / value);
                loadedKernel(meanValueBuffer.Length, meanValueBuffer, sourceBuffer.Length);

                var sumKernel = accelerator.LoadAutoGroupedStreamKernel((Index1 i, ArrayView<float> sourceData, ArrayView<float> destination, ArrayView<float> mean) => {
                    destination[i] = (sourceData[i] - mean[0]) * (sourceData[i] - mean[0]);
                });
                sumKernel(sumPoweredBuffer.Length, sourceBuffer, sumPoweredBuffer, meanValueBuffer);

                accelerator.Reduce<float, AddFloat>(accelerator.DefaultStream, sumPoweredBuffer.View, sumValueBuffer.View);

                accelerator.Synchronize();

                float sd = MathF.Sqrt(sumValueBuffer.GetAsArray().ElementAt(0) / Arr.Length);
                sourceBuffer.Dispose();
                meanValueBuffer.Dispose();
            }
            sw.Stop();
            Console.WriteLine($"GPU: {sw.ElapsedMilliseconds}");

            #endregion
        }
    }
}