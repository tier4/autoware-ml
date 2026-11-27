---
icon: lucide/route
---

# SparseDrive

SparseDrive (experimental) is a temporal lidar model built on sparse instances. `SparseDriveModel` runs the [PointTransformerV3](ptv3.md) encoder once per frame and feeds its stages to the heads:

- Detection and tracking: `PTv3Det3DFeaturePyramid` fuses the deepest encoder stage top-down into finer ones and feeds a `Sparse4DHead`. Its queries sample the sparse voxels and one dense BEV level at keypoints around their anchors, predict velocity, and carry the most confident instances from one frame of a scenario to the next.
- Segmentation: the PTv3 decoder head `PTv3Seg3DDecoderHead` labels every point.

The model streams the frames of every scenario in order, so it trains and runs on scenes at a steady frame rate, such as 10 Hz scenes, see `StreamingDataModule`.

## Summary

| Property     | Value                                                               |
| ------------ | ------------------------------------------------------------------- |
| Task         | 3D object detection and tracking, 3D semantic segmentation          |
| Modality     | LiDAR                                                               |
| Input        | Stream of single point clouds of x, y, z and intensity              |
| Output       | Tracked 3D boxes/scores/classes and point-wise semantic labels      |
| Architecture | PTv3 encoder, Sparse4D detection head and PTv3 segmentation decoder |
| Datasets     | T4Dataset                                                           |

## Available Configurations

| Config Name                                      | Task                         | Dataset   | Head     | Range | Purpose                                 |
| ------------------------------------------------ | ---------------------------- | --------- | -------- | ----- | --------------------------------------- |
| `multi/sparsedrive/t4dataset/base_voxel012_122m` | segmentation3d + detection3d | T4Dataset | Sparse4D | 122 m | Joint segmentation + temporal detection |

The rows are bases. They declare the classes the model trains and require the taxonomy that folds the labels of the database onto them, so an experiment that binds one lives in an external config root, see [Configuration](../user-guide/configuration.md#external-configs-and-scenario-lists).

The base samples the sparse voxels and one coarse encoder stage projected onto a BEV map. `model.bbox_head.lidar_sampling` selects `sparse`, `bev` or `sparse_bev`, and `model.bbox_head.bev` sets the projected stages. The top level `track_birth_score` is a training setting. In training, only a query scoring at least this score keeps its matched instance in its temporal slot, so the next frame matches that slot to the same object. At inference it is the score from which a query starts a track. The head, its export metadata, which the runtime reads, and every metric suite that scores the tracks take the one value, so an experiment sets it there and a deployment keeps the value it was trained with. The model summary of these experiments lists the modules without FLOPs, since a summary forward pass would feed the instance bank one frame outside its stream.

## Training

The Sparse4D head starts from k-means anchors of the training boxes. The `anchor_recipe` of every head sets how they are derived, and [Adding Models](../contributing/adding-models.md#key-points) describes how training builds them into the anchor cache and how weights carry them.

## Deployment

The `lidar_encoder` and `seg3d_head` graphs are the PTv3 ones, see [PointTransformerV3](ptv3.md#split-export-module-contract). The other graphs are:

- `det3d_head` - the temporal Sparse4D graph. It consumes the stage features from the finest pyramid stage to the deepest one with the pooling clusters and voxel coordinates in between, and takes the instance memory of the last frame (`temp_instance_feature`, `temp_anchor`, `temp_confidence`) with the continuity gate `temp_valid`, the frame gap `time_interval` and the transform `prev2current` from the last frame into the current one. The runtime opens the gate only for a frame gap within the `min_time_interval` and `max_time_interval` of the artifact metadata and passes `default_time_interval` otherwise. Besides the last decoder layer of every query (`classification`, `prediction`, `quality`) it returns the memory of the next frame (`temp_*_out`) and `temp_query_index_out`, the query every memory slot was selected from, which lets the runtime carry the track of a query into its slot. The runtime decodes every query into one box of its best class, scored by the class probability times the centerness, and keeps the top `num_output` boxes, nothing is suppressed, as `Sparse4DHead.decode_outputs` does for the metrics. Every decoder layer samples the stages with one `autoware::SparseDeformableAggregation` node, which reads the keypoints, the sampling weights and the padded features and dense int32 voxel lookups of every stage, and is served by the TensorRT plugin of the same name. With `lidar_sampling: sparse_bev` or `bev` the graph also projects the stages of `model.bbox_head.bev` onto dense BEV maps and reads them with one ONNX `GridSample` per level and decoder layer, and with `bev` it holds no plugin node.

## Implementation

| Path                                              | Description                                               |
| ------------------------------------------------- | --------------------------------------------------------- |
| `autoware_ml/models/sparsedrive/model.py`         | SparseDrive Lightning model wrapper                       |
| `autoware_ml/models/sparsedrive/lidar_stages.py`  | Fused PTv3 stages and BEV maps the detection head samples |
| `autoware_ml/models/sparsedrive/sparse4d_head.py` | Sparse4D temporal query detection head                    |
| `autoware_ml/anchors/`                            | Anchor recipes of the Sparse4D head                       |
| `autoware_ml/ops/deformable_aggregation/`         | Sparse4D keypoint sampling export operators               |
| `autoware_ml/datamodule/streaming_data_module.py` | Datamodule streaming the frames of every scenario         |
| `autoware_ml/configs/tasks/multi/sparsedrive/`    | SparseDrive experiment configurations                     |

## Acknowledgment

The Sparse4D head was ported from the official Sparse4D project by Horizon Robotics.

<!-- cspell:ignore Xuewu -->
- Repository: <https://github.com/HorizonRobotics/Sparse4D>
- License: MIT
- Paper: Lin, Xuewu, et al. "Sparse4D v3: Advancing End-to-End 3D Detection and Tracking." arXiv, 2023.
