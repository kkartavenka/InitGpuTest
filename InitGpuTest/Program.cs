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

            using var accelerator = new CudaAccelerator(context);
            var reduceFloat = accelerator.CreateReduction<float, AddFloat>();
            var loadedKernel = accelerator.LoadAutoGroupedStreamKernel((Index1 i, ArrayView<float> data, long value) => data[i] = data[i] / value);

            var sumKernel = accelerator.LoadAutoGroupedStreamKernel((Index1 i, ArrayView<float> sourceData, ArrayView<float> destination, ArrayView<float> mean) => {
                destination[i] = (sourceData[i] - mean[0]) * (sourceData[i] - mean[0]);
            });

            sw.Start();
            for (int i = 0; i < r; i++) {
                using var sourceBuffer = accelerator.Allocate(Arr);
                using var meanValueBuffer = accelerator.Allocate<float>(1);
                using var sumPoweredBuffer = accelerator.Allocate<float>(Arr.Length);
                using var sumValueBuffer = accelerator.Allocate<float>(1);

                reduceFloat(accelerator.DefaultStream, sourceBuffer.View, meanValueBuffer.View);

                loadedKernel(meanValueBuffer.Length, meanValueBuffer, sourceBuffer.Length);

                sumKernel(sumPoweredBuffer.Length, sourceBuffer, sumPoweredBuffer, meanValueBuffer);

                reduceFloat(accelerator.DefaultStream, sumPoweredBuffer.View, sumValueBuffer.View);

                sumValueBuffer.CopyTo(out float sumValueBuffer0, 0);
                float sd = MathF.Sqrt(sumValueBuffer0 / Arr.Length);
            }
            sw.Stop();
            Console.WriteLine($"GPU: {sw.ElapsedMilliseconds}");

            #endregion
        }
    }
}