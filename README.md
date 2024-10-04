2024-10-01 16:21:40 | INFO     | ONNXExporter | Starting ONNX to TensorRT conversion.
2024-10-01 16:21:41 | INFO     | ONNXExporter | Starting container nvcr.io/nvidia/tensorrt:22.10-py3 with GPU support and volume mapping.
2024-10-01 16:21:42 | INFO     | ONNXExporter | Docker container started successfully.
2024-10-01 16:21:42 | INFO     | ONNXExporter | Running TensorRT conversion command: trtexec --onnx=/workspace/yolov11m.onnx --saveEngine=/workspace/model.plan --fp16 --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640
2024-10-01 16:32:52 | INFO     | ONNXExporter | Conversion successful.
2024-10-01 16:32:53 | INFO     | ONNXExporter | &&&& RUNNING TensorRT.trtexec [TensorRT v8500] # trtexec --onnx=/workspace/yolov11m.onnx --saveEngine=/workspace/model.plan --fp16 --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640
[10/01/2024-13:21:44] [I] === Model Options ===
[10/01/2024-13:21:44] [I] Format: ONNX
[10/01/2024-13:21:44] [I] Model: /workspace/yolov11m.onnx
[10/01/2024-13:21:44] [I] Output:
[10/01/2024-13:21:44] [I] === Build Options ===
[10/01/2024-13:21:44] [I] Max batch: explicit batch
[10/01/2024-13:21:44] [I] Memory Pools: workspace: default, dlaSRAM: default, dlaLocalDRAM: default, dlaGlobalDRAM: default
[10/01/2024-13:21:44] [I] minTiming: 1
[10/01/2024-13:21:44] [I] avgTiming: 8
[10/01/2024-13:21:44] [I] Precision: FP32+FP16
[10/01/2024-13:21:44] [I] LayerPrecisions: 
[10/01/2024-13:21:44] [I] Calibration: 
[10/01/2024-13:21:44] [I] Refit: Disabled
[10/01/2024-13:21:44] [I] Sparsity: Disabled
[10/01/2024-13:21:44] [I] Safe mode: Disabled
[10/01/2024-13:21:44] [I] DirectIO mode: Disabled
[10/01/2024-13:21:44] [I] Restricted mode: Disabled
[10/01/2024-13:21:44] [I] Build only: Disabled
[10/01/2024-13:21:44] [I] Save engine: /workspace/model.plan
[10/01/2024-13:21:44] [I] Load engine: 
[10/01/2024-13:21:44] [I] Profiling verbosity: 0
[10/01/2024-13:21:44] [I] Tactic sources: Using default tactic sources
[10/01/2024-13:21:44] [I] timingCacheMode: local
[10/01/2024-13:21:44] [I] timingCacheFile: 
[10/01/2024-13:21:44] [I] Heuristic: Disabled
[10/01/2024-13:21:44] [I] Preview Features: Use default preview flags.
[10/01/2024-13:21:44] [I] Input(s)s format: fp32:CHW
[10/01/2024-13:21:44] [I] Output(s)s format: fp32:CHW
[10/01/2024-13:21:44] [I] Input build shape: images=1x3x640x640+4x3x640x640+8x3x640x640
[10/01/2024-13:21:44] [I] Input calibration shapes: model
[10/01/2024-13:21:44] [I] === System Options ===
[10/01/2024-13:21:44] [I] Device: 0
[10/01/2024-13:21:44] [I] DLACore: 
[10/01/2024-13:21:44] [I] Plugins:
[10/01/2024-13:21:44] [I] === Inference Options ===
[10/01/2024-13:21:44] [I] Batch: Explicit
[10/01/2024-13:21:44] [I] Input inference shape: images=4x3x640x640
[10/01/2024-13:21:44] [I] Iterations: 10
[10/01/2024-13:21:44] [I] Duration: 3s (+ 200ms warm up)
[10/01/2024-13:21:44] [I] Sleep time: 0ms
[10/01/2024-13:21:44] [I] Idle time: 0ms
[10/01/2024-13:21:44] [I] Streams: 1
[10/01/2024-13:21:44] [I] ExposeDMA: Disabled
[10/01/2024-13:21:44] [I] Data transfers: Enabled
[10/01/2024-13:21:44] [I] Spin-wait: Disabled
[10/01/2024-13:21:44] [I] Multithreading: Disabled
[10/01/2024-13:21:44] [I] CUDA Graph: Disabled
[10/01/2024-13:21:44] [I] Separate profiling: Disabled
[10/01/2024-13:21:44] [I] Time Deserialize: Disabled
[10/01/2024-13:21:44] [I] Time Refit: Disabled
[10/01/2024-13:21:44] [I] NVTX verbosity: 0
[10/01/2024-13:21:44] [I] Persistent Cache Ratio: 0
[10/01/2024-13:21:44] [I] Inputs:
[10/01/2024-13:21:44] [I] === Reporting Options ===
[10/01/2024-13:21:44] [I] Verbose: Disabled
[10/01/2024-13:21:44] [I] Averages: 10 inferences
[10/01/2024-13:21:44] [I] Percentiles: 90,95,99
[10/01/2024-13:21:44] [I] Dump refittable layers:Disabled
[10/01/2024-13:21:44] [I] Dump output: Disabled
[10/01/2024-13:21:44] [I] Profile: Disabled
[10/01/2024-13:21:44] [I] Export timing to JSON file: 
[10/01/2024-13:21:44] [I] Export output to JSON file: 
[10/01/2024-13:21:44] [I] Export profile to JSON file: 
[10/01/2024-13:21:44] [I] 
[10/01/2024-13:21:44] [I] === Device Information ===
[10/01/2024-13:21:44] [I] Selected Device: NVIDIA GeForce RTX 2080 Ti
[10/01/2024-13:21:44] [I] Compute Capability: 7.5
[10/01/2024-13:21:44] [I] SMs: 68
[10/01/2024-13:21:44] [I] Compute Clock Rate: 1.545 GHz
[10/01/2024-13:21:44] [I] Device Global Memory: 11011 MiB
[10/01/2024-13:21:44] [I] Shared Memory per SM: 64 KiB
[10/01/2024-13:21:44] [I] Memory Bus Width: 352 bits (ECC disabled)
[10/01/2024-13:21:44] [I] Memory Clock Rate: 7 GHz
[10/01/2024-13:21:44] [I] 
[10/01/2024-13:21:44] [I] TensorRT version: 8.5.0
[10/01/2024-13:21:45] [I] [TRT] [MemUsageChange] Init CUDA: CPU +306, GPU +0, now: CPU 319, GPU 312 (MiB)
[10/01/2024-13:21:47] [I] [TRT] [MemUsageChange] Init builder kernel library: CPU +260, GPU +74, now: CPU 632, GPU 386 (MiB)
[10/01/2024-13:21:47] [W] [TRT] CUDA lazy loading is not enabled. Enabling it can significantly reduce device memory usage. See `CUDA_MODULE_LOADING` in https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
[10/01/2024-13:21:47] [I] Start parsing network model
[10/01/2024-13:21:47] [I] [TRT] ----------------------------------------------------------------
[10/01/2024-13:21:47] [I] [TRT] Input filename:   /workspace/yolov11m.onnx
[10/01/2024-13:21:47] [I] [TRT] ONNX IR version:  0.0.10
[10/01/2024-13:21:47] [I] [TRT] Opset version:    19
[10/01/2024-13:21:47] [I] [TRT] Producer name:    pytorch
[10/01/2024-13:21:47] [I] [TRT] Producer version: 2.4.1
[10/01/2024-13:21:47] [I] [TRT] Domain:           
[10/01/2024-13:21:47] [I] [TRT] Model version:    0
[10/01/2024-13:21:47] [I] [TRT] Doc string:       
[10/01/2024-13:21:47] [I] [TRT] ----------------------------------------------------------------
[10/01/2024-13:21:47] [W] [TRT] parsers/onnx/onnx2trt_utils.cpp:375: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
[10/01/2024-13:21:47] [I] Finish parsing network model
[10/01/2024-13:21:49] [I] [TRT] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +463, GPU +192, now: CPU 1192, GPU 586 (MiB)
[10/01/2024-13:21:49] [I] [TRT] [MemUsageChange] Init cuDNN: CPU +115, GPU +52, now: CPU 1307, GPU 638 (MiB)
[10/01/2024-13:21:49] [I] [TRT] Local timing cache in use. Profiling results in this builder pass will not be stored.
[10/01/2024-13:25:26] [W] [TRT] Cache result detected as invalid for node: /model.9/m_2/MaxPool, LayerImpl: CaskPooling, tactic: 0x457fb0a334d63ae2
[10/01/2024-13:27:02] [I] [TRT] Detected 1 inputs and 3 output network tensors.
[10/01/2024-13:27:02] [I] [TRT] Total Host Persistent Memory: 288992
[10/01/2024-13:27:02] [I] [TRT] Total Device Persistent Memory: 1981952
[10/01/2024-13:27:02] [I] [TRT] Total Scratch Memory: 0
[10/01/2024-13:27:02] [I] [TRT] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 40 MiB, GPU 8974 MiB
[10/01/2024-13:27:02] [I] [TRT] [BlockAssignment] Algorithm ShiftNTopDown took 70.8212ms to assign 13 blocks to 219 nodes requiring 273613832 bytes.
[10/01/2024-13:27:02] [I] [TRT] Total Activation Memory: 273613832
[10/01/2024-13:27:02] [I] [TRT] [MemUsageChange] Init cuDNN: CPU +0, GPU +10, now: CPU 1828, GPU 882 (MiB)
[10/01/2024-13:27:02] [W] [TRT] TensorRT encountered issues when converting weights between types and that could affect accuracy.
[10/01/2024-13:27:02] [W] [TRT] If this is not the desired behavior, please modify the weights or retrain with regularization to adjust the magnitude of the weights.
[10/01/2024-13:27:02] [W] [TRT] Check verbose logs for the list of affected weights.
[10/01/2024-13:27:02] [W] [TRT] - 103 weights are affected by this issue: Detected subnormal FP16 values.
[10/01/2024-13:27:02] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in building engine: CPU +39, GPU +40, now: CPU 39, GPU 40 (MiB)
[10/01/2024-13:27:02] [I] Engine built in 318.021 sec.
[10/01/2024-13:27:02] [I] [TRT] Loaded engine size: 40 MiB
[10/01/2024-13:27:02] [I] [TRT] [MemUsageChange] Init cuDNN: CPU +0, GPU +10, now: CPU 1485, GPU 804 (MiB)
[10/01/2024-13:27:02] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +40, now: CPU 0, GPU 40 (MiB)
[10/01/2024-13:27:02] [I] Engine deserialized in 0.0326257 sec.
[10/01/2024-13:27:02] [I] [TRT] [MemUsageChange] Init cuDNN: CPU +0, GPU +8, now: CPU 1486, GPU 806 (MiB)
[10/01/2024-13:27:02] [W] [TRT] CUDA lazy loading is not enabled. Enabling it can significantly reduce device memory usage. See `CUDA_MODULE_LOADING` in https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
[10/01/2024-13:27:02] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +263, now: CPU 0, GPU 303 (MiB)
[10/01/2024-13:27:02] [I] Setting persistentCacheLimit to 0 bytes.
[10/01/2024-13:27:02] [I] Using random values for input images
[10/01/2024-13:27:03] [I] Created input binding for images with dimensions 4x3x640x640
[10/01/2024-13:27:03] [I] Using random values for output output0
[10/01/2024-13:27:03] [I] Created output binding for output0 with dimensions 4x84x8400
[10/01/2024-13:27:03] [I] Starting inference
[10/01/2024-13:27:06] [I] Warmup completed 28 queries over 200 ms
[10/01/2024-13:27:06] [I] Timing trace has 410 queries over 3.02161 s
[10/01/2024-13:27:06] [I] 
[10/01/2024-13:27:06] [I] === Trace details ===
[10/01/2024-13:27:06] [I] Trace averages of 10 runs:
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.28331 ms - Host latency: 9.78616 ms (enqueue 1.30835 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.31797 ms - Host latency: 9.82058 ms (enqueue 1.2998 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.30121 ms - Host latency: 9.8033 ms (enqueue 1.27233 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.27156 ms - Host latency: 9.76617 ms (enqueue 1.26566 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.34681 ms - Host latency: 9.84652 ms (enqueue 1.35998 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.28901 ms - Host latency: 9.78999 ms (enqueue 1.27892 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.28663 ms - Host latency: 9.78557 ms (enqueue 1.31749 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.29865 ms - Host latency: 9.80212 ms (enqueue 1.27586 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.25928 ms - Host latency: 9.75422 ms (enqueue 1.32923 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.35859 ms - Host latency: 9.85643 ms (enqueue 1.28542 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.3587 ms - Host latency: 9.8573 ms (enqueue 1.32992 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.29194 ms - Host latency: 9.78958 ms (enqueue 1.34174 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.28684 ms - Host latency: 9.78763 ms (enqueue 1.36134 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.2998 ms - Host latency: 9.79867 ms (enqueue 1.28751 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.23649 ms - Host latency: 9.72839 ms (enqueue 1.37578 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.32729 ms - Host latency: 9.82485 ms (enqueue 1.35154 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.33502 ms - Host latency: 9.83679 ms (enqueue 1.30608 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.33231 ms - Host latency: 9.83311 ms (enqueue 1.35973 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.29429 ms - Host latency: 9.79397 ms (enqueue 1.29397 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.3264 ms - Host latency: 9.82222 ms (enqueue 1.13252 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.28718 ms - Host latency: 9.78649 ms (enqueue 1.21685 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.39557 ms - Host latency: 9.90116 ms (enqueue 1.25012 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.36726 ms - Host latency: 9.87059 ms (enqueue 1.35223 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.35522 ms - Host latency: 9.85743 ms (enqueue 1.34802 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.31547 ms - Host latency: 9.81656 ms (enqueue 1.36642 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.33381 ms - Host latency: 9.83654 ms (enqueue 1.42034 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.32703 ms - Host latency: 9.82683 ms (enqueue 1.2645 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.33293 ms - Host latency: 9.83184 ms (enqueue 1.35762 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.35146 ms - Host latency: 9.85215 ms (enqueue 1.33872 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.31404 ms - Host latency: 9.81743 ms (enqueue 1.29753 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.35864 ms - Host latency: 9.85999 ms (enqueue 1.35154 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.32095 ms - Host latency: 9.81885 ms (enqueue 1.29243 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.32578 ms - Host latency: 9.82539 ms (enqueue 1.39197 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.33469 ms - Host latency: 9.83662 ms (enqueue 1.32568 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.3686 ms - Host latency: 9.86856 ms (enqueue 1.26868 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.32788 ms - Host latency: 9.82847 ms (enqueue 1.30271 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.37122 ms - Host latency: 9.87229 ms (enqueue 1.25476 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.34409 ms - Host latency: 9.8446 ms (enqueue 1.34783 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.3197 ms - Host latency: 9.8218 ms (enqueue 1.33259 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.34622 ms - Host latency: 9.84602 ms (enqueue 1.30027 ms)
[10/01/2024-13:27:06] [I] Average on 10 runs - GPU latency: 7.30942 ms - Host latency: 9.8114 ms (enqueue 1.40342 ms)
[10/01/2024-13:27:06] [I] 
[10/01/2024-13:27:06] [I] === Performance summary ===
[10/01/2024-13:27:06] [I] Throughput: 135.689 qps
[10/01/2024-13:27:06] [I] Latency: min = 9.5257 ms, max = 10.0364 ms, mean = 9.82231 ms, median = 9.82507 ms, percentile(90%) = 9.88892 ms, percentile(95%) = 9.9043 ms, percentile(99%) = 9.96838 ms
[10/01/2024-13:27:06] [I] Enqueue Time: min = 0.921631 ms, max = 2.23669 ms, mean = 1.31506 ms, median = 1.32135 ms, percentile(90%) = 1.61548 ms, percentile(95%) = 1.71899 ms, percentile(99%) = 1.81921 ms
[10/01/2024-13:27:06] [I] H2D Latency: min = 1.6131 ms, max = 1.67749 ms, mean = 1.63842 ms, median = 1.63818 ms, percentile(90%) = 1.64844 ms, percentile(95%) = 1.65234 ms, percentile(99%) = 1.65768 ms
[10/01/2024-13:27:06] [I] GPU Compute Time: min = 7.02802 ms, max = 7.53619 ms, mean = 7.32218 ms, median = 7.32544 ms, percentile(90%) = 7.38696 ms, percentile(95%) = 7.3988 ms, percentile(99%) = 7.46777 ms
[10/01/2024-13:27:06] [I] D2H Latency: min = 0.85791 ms, max = 0.874146 ms, mean = 0.861711 ms, median = 0.861511 ms, percentile(90%) = 0.862793 ms, percentile(95%) = 0.863159 ms, percentile(99%) = 0.869995 ms
[10/01/2024-13:27:06] [I] Total Host Walltime: 3.02161 s
[10/01/2024-13:27:06] [I] Total GPU Compute Time: 3.00209 s
[10/01/2024-13:27:06] [I] Explanations of the performance metrics are printed in the verbose logs.
[10/01/2024-13:27:06] [I] 
&&&& PASSED TensorRT.trtexec [TensorRT v8500] # trtexec --onnx=/workspace/yolov11m.onnx --saveEngine=/workspace/model.plan --fp16 --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640

2024-10-01 16:32:57 | INFO     | ONNXExporter | Stopping and removing Docker container.