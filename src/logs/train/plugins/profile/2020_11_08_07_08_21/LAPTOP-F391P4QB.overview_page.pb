�	xz�,#c�@xz�,#c�@!xz�,#c�@	DM6��j?DM6��j?!DM6��j?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$xz�,#c�@��6�[�?A��u��b�@Y;�O��n�?*	�����<c@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���(\��?!�Cθ�G@)~��k	��?1����:�E@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat8gDio�?!���6��4@)S�!�uq�?1J�ү�i1@:Preprocessing2U
Iterator::Model::ParallelMapV2�l����?!U��W�
(@)�l����?1U��W�
(@:Preprocessing2F
Iterator::Model�V-�?!�۝"7@)�� �rh�?1$=۞&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���_vO�?!��y�;S@)46<�R�?1�ouj�T@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicea��+ey?!*W��F@)a��+ey?1*W��F@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�g��s�u?!xc�8�@)�g��s�u?1xc�8�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapvOjM�?!Fp�H@)�����g?1�7;4B&�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9DM6��j?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��6�[�?��6�[�?!��6�[�?      ��!       "      ��!       *      ��!       2	��u��b�@��u��b�@!��u��b�@:      ��!       B      ��!       J	;�O��n�?;�O��n�?!;�O��n�?R      ��!       Z	;�O��n�?;�O��n�?!;�O��n�?JCPU_ONLYYDM6��j?b Y      Y@qS%�����?"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 