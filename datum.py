import abc
import dataclasses as dc
import pickle as pkl
import time

import lmdb
import numpy as np
import torch as pt
import torch.utils.data as ptud
import torchvision.transforms as ptvt
import zstd as zs

from utils import register_module


#### transforms


import tensorflow as tf
import tensorflow.python.framework.ops as tfpfo

tf.config.set_visible_devices([], "GPU")
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != "GPU"

SEED_KEY = "seed"
NOTRACK_BOX = (0.0, 0.0, 0.0, 0.0)  # No-track bounding box for padding.
NOTRACK_LABEL = -1

IMAGE = "image"
VIDEO = "video"
SEGMENTATIONS = "segmentations"
RAGGED_SEGMENTATIONS = "ragged_segmentations"
SPARSE_SEGMENTATIONS = "sparse_segmentations"
SHAPE = "shape"
PADDING_MASK = "padding_mask"
RAGGED_BOXES = "ragged_boxes"
BOXES = "boxes"
COND_MASK_KEY = "cond_mask"  # XXX
FRAMES = "frames"
FLOW = "flow"
DEPTH = "depth"
ORIGINAL_SIZE = "original_size"
INSTANCE_LABELS = "instance_labels"
INSTANCE_MULTI_LABELS = "instance_multi_labels"


def convert_uint16_to_float(array, min_val, max_val):
    return tf.cast(array, tf.float32) / 65535.0 * (max_val - min_val) + min_val


def adjust_small_size(original_size, small_size: int, max_size: int) -> int:
    """Computes the adjusted small size to ensure large side < max_size."""
    h, w = original_size
    min_original_size = tf.cast(tf.minimum(w, h), tf.float32)
    max_original_size = tf.cast(tf.maximum(w, h), tf.float32)
    if max_original_size / min_original_size * small_size > max_size:
        small_size = tf.cast(
            tf.floor(max_size * min_original_size / max_original_size), tf.int32
        )
    return small_size


def get_resize_small_shape(original_size, small_size: int):
    h, w = original_size
    ratio = tf.cast(small_size, tf.float32) / tf.cast(tf.minimum(h, w), tf.float32)
    h = tf.cast(tf.round(tf.cast(h, tf.float32) * ratio), tf.int32)
    w = tf.cast(tf.round(tf.cast(w, tf.float32) * ratio), tf.int32)
    return h, w


def flow_tensor_to_rgb_tensor(motion_image, flow_scaling_factor=50.0):
    """Visualizes flow motion image as an RGB image.

    Similar as the flow_to_rgb function, but with tensors.

    Args:
      motion_image: A tensor either of shape [batch_sz, height, width, 2] or of
        shape [height, width, 2]. motion_image[..., 0] is flow in x and
        motion_image[..., 1] is flow in y.
      flow_scaling_factor: How much to scale flow for visualization.

    Returns:
      A visualization tensor with same shape as motion_image, except with three
      channels. The dtype of the output is tf.uint8.
    """
    hypot = lambda a, b: (a**2.0 + b**2.0) ** 0.5  # sqrt(a^2 + b^2)

    height, width = motion_image.get_shape().as_list()[
        -3:-1
    ]  # pytype: disable=attribute-error  # allow-recursive-types
    scaling = flow_scaling_factor / hypot(height, width)
    x, y = motion_image[..., 0], motion_image[..., 1]
    motion_angle = tf.atan2(y, x)
    motion_angle = (motion_angle / np.math.pi + 1.0) / 2.0
    motion_magnitude = hypot(y, x)
    motion_magnitude = tf.clip_by_value(motion_magnitude * scaling, 0.0, 1.0)
    value_channel = tf.ones_like(motion_angle)
    flow_hsv = tf.stack([motion_angle, motion_magnitude, value_channel], axis=-1)
    flow_rgb = tf.image.convert_image_dtype(tf.image.hsv_to_rgb(flow_hsv), tf.uint8)
    return flow_rgb


@dc.dataclass
class VideoFromTfds:
    """Standardize features coming from TFDS video datasets."""

    video_key: str = VIDEO
    segmentations_key: str = SEGMENTATIONS
    ragged_segmentations_key: str = RAGGED_SEGMENTATIONS
    shape_key: str = SHAPE
    padding_mask_key: str = PADDING_MASK
    ragged_boxes_key: str = RAGGED_BOXES
    boxes_key: str = BOXES
    frames_key: str = FRAMES
    instance_multi_labels_key: str = INSTANCE_MULTI_LABELS
    flow_key: str = FLOW
    depth_key: str = DEPTH

    def __call__(self, features):
        features_new = {}

        if "rng" in features:
            features_new[SEED_KEY] = features.pop("rng")

        if "instances" in features:
            features_new[self.ragged_boxes_key] = features["instances"]["bboxes"]
            features_new[self.frames_key] = features["instances"]["bbox_frames"]
            if "segmentations" in features["instances"]:
                features_new[self.ragged_segmentations_key] = tf.cast(
                    features["instances"]["segmentations"][..., 0], tf.int32
                )

            # Special handling of CLEVR (https://arxiv.org/abs/1612.06890) objects.
            if (
                "color" in features["instances"]
                and "shape" in features["instances"]
                and "material" in features["instances"]
            ):
                color = tf.cast(features["instances"]["color"], tf.int32)
                shape = tf.cast(features["instances"]["shape"], tf.int32)
                material = tf.cast(features["instances"]["material"], tf.int32)
                features_new[self.instance_multi_labels_key] = tf.stack(
                    (color, shape, material), axis=-1
                )

        if "segmentations" in features:
            features_new[self.segmentations_key] = tf.cast(
                features["segmentations"][..., 0], tf.int32
            )

        if "depth" in features:
            # Undo float to uint16 scaling
            depth_range = features["metadata"]["depth_range"]
            features_new[self.depth_key] = convert_uint16_to_float(
                features["depth"], depth_range[0], depth_range[1]
            )

        if "flows" in features:
            # Some datasets use "flows" instead of "flow" for optical flow.
            features["flow"] = features["flows"]
        if "backward_flow" in features:
            # By default, use "backward_flow" if available.
            features["flow"] = features["backward_flow"]
            features["metadata"]["flow_range"] = features["metadata"][
                "backward_flow_range"
            ]
        if "flow" in features:
            # Undo float to uint16 scaling
            flow_range = features["metadata"].get("flow_range", (-255, 255))
            features_new[self.flow_key] = convert_uint16_to_float(
                features["flow"], flow_range[0], flow_range[1]
            )

        # Convert video to float and normalize.
        video = features["video"]
        assert (
            video.dtype == tf.uint8
        )  # pytype: disable=attribute-error  # allow-recursive-types
        video = tf.image.convert_image_dtype(video, tf.float32)
        features_new[self.video_key] = video

        # Store original video shape (e.g. for correct evaluation metrics).
        features_new[self.shape_key] = tf.shape(video)

        # Store padding mask
        features_new[self.padding_mask_key] = tf.cast(
            tf.ones_like(video)[..., 0], tf.uint8
        )

        return features_new


@dc.dataclass
class SparseToDenseAnnotation:
    """Converts the sparse to a dense representation.

    Returns the following fields:
      - `video`: A dense tensor of shape (number of frames, height, width, 3).
      - `boxes`: Converts the tracks to a dense tensor of shape
        (number of annotated frames, `max_instances` tracks, 4).
      - `segmentations`: If sparse segmentations are available, they are converted
        to a dense segmentation tensor of shape (#frames, height, width, 1) with
        integers reaching from 0 (background) to `max_instances`.
    """

    max_instances: int = 10

    video_key: str = VIDEO
    ragged_boxes_key: str = RAGGED_BOXES
    cond_mask_key: str = COND_MASK_KEY
    boxes_key: str = BOXES
    frames_key: str = FRAMES
    ragged_segmentations_key: str = RAGGED_SEGMENTATIONS
    segmentations_key: str = SEGMENTATIONS
    padding_mask_key: str = PADDING_MASK
    instance_labels_key: str = INSTANCE_LABELS
    instance_multi_labels_key: str = INSTANCE_MULTI_LABELS

    def __call__(self, features):
        def crop_or_pad(t, size, value, allow_crop=True):
            pad_size = tf.maximum(size - tf.shape(t)[0], 0)
            t = tf.pad(
                t,
                ((0, pad_size),)
                + ((0, 0),)
                * (
                    len(t.shape) - 1
                ),  # pytype: disable=attribute-error  # allow-recursive-types
                constant_values=value,
            )
            if allow_crop:
                t = t[:size]
            return t

        updated_keys = {
            self.video_key,
            self.frames_key,
            self.ragged_boxes_key,
            self.ragged_segmentations_key,
            self.segmentations_key,
            self.instance_labels_key,
            self.instance_multi_labels_key,
        }
        features_new = {k: v for k, v in features.items() if k not in updated_keys}

        frames = features[self.frames_key]
        frames = tf.ragged.constant(frames)  # XXX
        frames_dense = frames.to_tensor(
            default_value=0
        )  # pytype: disable=attribute-error  # allow-recursive-types
        video = features[self.video_key]
        features_new[self.video_key] = video
        num_frames = tf.shape(video)[0]
        num_tracks = tf.shape(frames_dense)[0]

        # Densify segmentations.
        if self.ragged_segmentations_key in features:
            segmentations = features[self.ragged_segmentations_key]
            dense_segmentations = tf.zeros_like(
                features[self.padding_mask_key], tf.int32
            )

            def densify_segmentations(dense_segmentations, vals):
                """Densify non-overlapping segmentations."""
                frames, segmentations, idx = vals
                return tf.tensor_scatter_nd_add(
                    dense_segmentations, frames[:, tf.newaxis], segmentations * idx
                )

            # We can safely convert the RaggedTensors to dense as all zero values are
            # ignored due to the aggregation via scatter_nd_add. We also crop to the
            # maximum number of tracks.
            scan_tuple = (
                crop_or_pad(frames_dense, self.max_instances, 0),
                crop_or_pad(
                    segmentations.to_tensor(
                        default_value=0
                    ),  # pytype: disable=attribute-error  # allow-recursive-types
                    self.max_instances,
                    0,
                ),
                tf.range(1, self.max_instances + 1),
            )

            features_new[self.segmentations_key] = tf.scan(
                densify_segmentations, scan_tuple, dense_segmentations
            )[-1]
        elif self.segmentations_key in features:
            # Dense segmentations are available for this dataset. It may be that
            # max_instances < max(features_new[self.segmentations_key]). We prune out
            # extra objects here.
            segmentations = features[self.segmentations_key]
            segmentations = tf.where(
                tf.less_equal(segmentations, self.max_instances), segmentations, 0
            )
            features_new[self.segmentations_key] = segmentations

        # Densify boxes.
        bboxes = features[self.ragged_boxes_key]

        def densify_boxes(n):
            boxes_n = tf.tensor_scatter_nd_update(
                tf.tile(tf.constant(NOTRACK_BOX)[tf.newaxis], (num_frames, 1)),
                frames[n][:, tf.newaxis],
                bboxes[n],
            )
            return boxes_n

        boxes0 = tf.map_fn(
            densify_boxes,
            tf.range(tf.minimum(num_tracks, self.max_instances)),
            fn_output_signature=tf.float32,
        )
        boxes = tf.transpose(
            crop_or_pad(boxes0, self.max_instances, NOTRACK_BOX[0]), (1, 0, 2)
        )
        features_new[self.boxes_key] = tf.ensure_shape(
            boxes, (None, self.max_instances, len(NOTRACK_BOX))
        )
        cond_mask = tf.ones([num_tracks + 1, num_frames], dtype=tf.uint8)
        cond_mask = tf.transpose(
            crop_or_pad(cond_mask, self.max_instances + 1, 0), (1, 0)
        )
        features_new[self.cond_mask_key] = cond_mask

        # Labels.
        if self.instance_labels_key in features:
            labels = crop_or_pad(
                features[self.instance_labels_key], self.max_instances, NOTRACK_LABEL
            )
            features_new[self.instance_labels_key] = tf.ensure_shape(
                labels, (self.max_instances,)
            )

        # Multi-labels.
        if self.instance_multi_labels_key in features:
            multi_labels = crop_or_pad(
                features[self.instance_multi_labels_key],
                self.max_instances,
                NOTRACK_LABEL,
            )
            features_new[self.instance_multi_labels_key] = tf.ensure_shape(
                multi_labels, (self.max_instances, multi_labels.get_shape()[1])
            )

        # Frames.
        features_new[self.frames_key] = frames

        return features_new


# @dc.dataclass
class VideoPreprocessOp(abc.ABC):
    """Base class for all video preprocess ops."""

    video_key: str = VIDEO
    segmentations_key: str = SEGMENTATIONS
    padding_mask_key: str = PADDING_MASK
    boxes_key: str = BOXES
    cond_mask_key: str = COND_MASK_KEY
    flow_key: str = FLOW
    depth_key: str = DEPTH
    sparse_segmentations_key: str = SPARSE_SEGMENTATIONS

    def __call__(self, features):
        # Get current video shape.
        video_shape = tf.shape(features[self.video_key])
        # Assemble all feature keys that the op should be applied on.
        all_keys = [
            self.video_key,
            self.segmentations_key,
            self.padding_mask_key,
            self.flow_key,
            self.depth_key,
            self.sparse_segmentations_key,
            self.boxes_key,
            self.cond_mask_key,
        ]
        # Apply the op to all features.
        for key in all_keys:
            if key in features:
                features[key] = self.apply(features[key], key, video_shape)
        return features

    @abc.abstractmethod
    def apply(self, tensor: tf.Tensor, key: str, video_shape: tf.TensorShape):
        pass


# @dc.dataclass
class RandomVideoPreprocessOp(VideoPreprocessOp):
    """Base class for all random video preprocess ops."""

    def __call__(self, features):
        if features.get(SEED_KEY) is None:
            op_seed = tf.random.uniform(shape=(2,), maxval=2**32, dtype=tf.int64)
        else:
            features[SEED_KEY], op_seed = tf.unstack(
                tf.random.experimental.stateless_split(features[SEED_KEY])
            )
        # Get current video shape.
        video_shape = tf.shape(features[self.video_key])
        # Assemble all feature keys that the op should be applied on.
        all_keys = [
            self.video_key,
            self.segmentations_key,
            self.padding_mask_key,
            self.flow_key,
            self.depth_key,
            self.sparse_segmentations_key,
            self.boxes_key,
            self.cond_mask_key,
        ]
        # Apply the op to all features.
        for key in all_keys:
            if key in features:
                features[key] = self.apply(features[key], op_seed, key, video_shape)
        return features

    @abc.abstractmethod
    def apply(
        self, tensor: tf.Tensor, seed: tf.Tensor, key: str, video_shape: tf.TensorShape
    ):
        pass


@dc.dataclass
class TemporalRandomStridedWindow(RandomVideoPreprocessOp):
    """Gets a random strided slice (window) along 0-th axis of input tensor."""

    length: int

    def _apply(self, tensor: tf.Tensor, seed, constant_values):
        """Applies the strided crop operation to the video tensor."""
        num_frames = tf.shape(tensor)[0]
        # XXX 反复确认了，没有seed就是不行（跟jax.jit无关），arifg~=60%。
        #   有seed时，interval=1较6更易恶化arifg<70%。
        mode = 1
        if mode == 1:  # ``random_strided_window=original__interval=6``
            num_crop_points = tf.cast(tf.math.ceil(num_frames / self.length), tf.int32)
            crop_point = tf.random.stateless_uniform(
                shape=(), minval=0, maxval=num_crop_points, dtype=tf.int32, seed=seed
            )
            crop_point *= self.length
        elif mode == 2:  # ``random_strided_window=mine__interval=1``
            crop_point = tf.random.stateless_uniform(
                shape=(),
                minval=0,
                maxval=num_frames - self.length + 1,
                dtype=tf.int32,
                seed=seed,
            )
        elif mode == 3:  # ``random_strided_window=mine__interval=6_noseed``
            num_crop_points = tf.cast(tf.math.ceil(num_frames / self.length), tf.int32)
            crop_point = tf.random.uniform(
                shape=(), minval=0, maxval=num_crop_points, dtype=tf.int32
            )
            crop_point *= self.length
        elif mode == 4:  # ``random_strided_window=mine__interval=1_noseed``
            crop_point = tf.random.uniform(
                shape=(), minval=0, maxval=num_frames - self.length + 1, dtype=tf.int32
            )
        else:
            raise "NotImplemented"
        frames_sample = tensor[crop_point : crop_point + self.length]
        frames_to_pad = tf.maximum(self.length - tf.shape(frames_sample)[0], 0)
        frames_sample = tf.pad(
            frames_sample,
            ((0, frames_to_pad),) + ((0, 0),) * (len(frames_sample.shape) - 1),
            constant_values=constant_values,
        )
        frames_sample = tf.ensure_shape(
            frames_sample, [self.length] + frames_sample.get_shape()[1:]
        )
        return frames_sample

    def apply(self, tensor, seed, key=None, video_shape=None):
        """See base class."""
        del video_shape
        if key == self.boxes_key:
            constant_values = NOTRACK_BOX[0]
        else:
            constant_values = 0
        return self._apply(tensor, seed, constant_values=constant_values)


@dc.dataclass
class TemporalCropOrPad(VideoPreprocessOp):
    """Crops or pads a video in time to a specified length."""

    length: int
    allow_crop: bool = True

    def _apply(self, tensor, constant_values):
        frames_to_pad = self.length - tf.shape(tensor)[0]
        if self.allow_crop:
            frames_to_pad = tf.maximum(frames_to_pad, 0)
        tensor = tf.pad(
            tensor,
            ((0, frames_to_pad),) + ((0, 0),) * (len(tensor.shape) - 1),
            constant_values=constant_values,
        )
        tensor = tensor[: self.length]
        tensor = tf.ensure_shape(tensor, [self.length] + tensor.get_shape()[1:])
        return tensor

    def apply(self, tensor, key=None, video_shape=None):
        """See base class."""
        del video_shape
        if key == self.boxes_key:
            constant_values = NOTRACK_BOX[0]
        else:
            constant_values = 0
        return self._apply(tensor, constant_values=constant_values)


@dc.dataclass
class ResizeSmall(VideoPreprocessOp):
    """Resizes the smaller spatial side to `size` keeping aspect ratio."""

    size: int
    max_size: int = None

    def apply(self, tensor, key=None, video_shape=None):
        """See base class."""

        # Boxes are defined in normalized image coordinates and are not affected.
        if key in [self.boxes_key, self.cond_mask_key]:
            return tensor

        if key in (self.padding_mask_key, self.segmentations_key):
            tensor = tensor[..., tf.newaxis]
        elif key == self.sparse_segmentations_key:
            tensor = tf.reshape(
                tensor, (-1, tf.shape(tensor)[2], tf.shape(tensor)[3], 1)
            )

        h, w = tf.shape(tensor)[1], tf.shape(tensor)[2]

        # Determine resize method based on dtype (e.g. segmentations are int).
        if tensor.dtype.is_integer:
            resize_method = "nearest"
        else:
            resize_method = "bilinear"

        # Clip size to max_size if needed.
        small_size = self.size
        if self.max_size is not None:
            small_size = adjust_small_size(
                original_size=(h, w), small_size=small_size, max_size=self.max_size
            )
        new_h, new_w = get_resize_small_shape(
            original_size=(h, w), small_size=small_size
        )
        tensor = tf.image.resize(tensor, [new_h, new_w], method=resize_method)

        # Flow needs to be rescaled according to the new size to stay valid.
        if key == self.flow_key:
            scale_h = tf.cast(new_h, tf.float32) / tf.cast(h, tf.float32)
            scale_w = tf.cast(new_w, tf.float32) / tf.cast(w, tf.float32)
            scale = tf.reshape(tf.stack([scale_h, scale_w], axis=0), (1, 2))
            # Optionally repeat scale in case both forward and backward flow are
            # stacked in the last dimension.
            scale = tf.repeat(scale, tf.shape(tensor)[-1] // 2, axis=0)
            scale = tf.reshape(scale, (1, 1, 1, tf.shape(tensor)[-1]))
            tensor *= scale

        if key in (self.padding_mask_key, self.segmentations_key):
            tensor = tensor[..., 0]
        elif key == self.sparse_segmentations_key:
            tensor = tf.reshape(tensor, (video_shape[0], -1, new_h, new_w))

        return tensor


@dc.dataclass
class FlowToRgb:
    """Converts flow to an RGB image."""

    flow_key: str = FLOW

    def __call__(self, features):
        if self.flow_key in features:
            flow_rgb = flow_tensor_to_rgb_tensor(features[self.flow_key])
            assert flow_rgb.dtype == tf.uint8
            features[self.flow_key] = tf.image.convert_image_dtype(flow_rgb, tf.float32)
        return features


#### datasets


# https://gregoryszorc.com/blog/2017/03/07/better-compression-with-zstandard/
compress = lambda _: zs.compress(pkl.dumps(_, pkl.HIGHEST_PROTOCOL))
decompress = lambda _: pkl.loads(zs.decompress(_))


class MoviLmdb(ptud.Dataset):
    """"""

    def __init__(self, data_file, transform, max_spare=4):
        super().__init__()
        self.env = lmdb.open(
            str(data_file),
            subdir=False,
            readonly=True,
            readahead=False,
            meminit=False,
            max_spare_txns=max_spare,
            lock=False,
        )
        with self.env.begin(write=False) as txn:
            self.keys = decompress(txn.get(b"__keys__"))
        self.transform = ptvt.Compose(transform)

    def __getitem__(
        self, index, permute_keys=("depth", "flow", "video"), pop_keys=("frames",)
    ):
        with self.env.begin(write=False) as txn:
            sample = decompress(txn.get(self.keys[index]))
        sample = self.transform(sample)
        sample2 = {}
        for key, value in sample.items():
            if key in pop_keys:
                continue
            if key in permute_keys:
                assert len(value.shape) == 4
                value = np.transpose(value, [0, 3, 1, 2])
            if key == "segmentations":
                value = tf.cast(value, dtype=tf.int64)
            sample2[key] = value
        return sample2

    def __len__(self):
        return len(self.keys)

    @staticmethod
    def collate_fn(samples: list) -> dict:
        assert isinstance(samples, list) and isinstance(samples[0], dict)
        batch = {}
        for key, value in samples[0].items():
            values = [_[key] for _ in samples]
            if isinstance(value, dict):
                values2 = MoviLmdb.collate_fn(values)
            elif isinstance(value, (np.ndarray, tfpfo.EagerTensor)):  # XXX
                values2 = pt.from_numpy(np.stack(values))
            elif isinstance(value, list):
                values2 = values
            else:
                print(type(value))
                raise "NotImplemented"
            batch[key] = values2
        return batch


DataLoader = ptud.DataLoader


[register_module(_) for _ in locals().values() if isinstance(_, type)]


####


def main_convert_tfds_to_lmdb():
    """
    Convert the original TFRecord files into one LMDB file, saving 80% storage space.
    """
    import time
    import pathlib as pl

    from clu import deterministic_data
    import tensorflow as tf
    import tensorflow.python.framework.ops as tfpfo
    import tensorflow_datasets as tfds

    _gpus = tf.config.list_physical_devices("GPU")
    [tf.config.experimental.set_memory_growth(_, True) for _ in _gpus]

    data_dir = "/media/GeneralZ/Storage/Static/datasets/tfds"  # XXX
    tfds_name = "movi_a/128x128:1.0.0"  # XXX
    save_dir = pl.Path("./")  # XXX
    write_freq = 64

    def _convert_nested_mapping(mapping: dict):
        mapping2 = {}
        for key, value in mapping.items():
            if isinstance(value, dict):
                value2 = _convert_nested_mapping(value)
            elif isinstance(value, tfpfo.EagerTensor):
                value2 = value.numpy()
            elif isinstance(value, tf.RaggedTensor):
                value2 = value.to_list()
            else:
                raise "NotImplemented"
            mapping2[key] = value2
        return mapping2

    for split in ["train", "validation"]:
        print(split)

        dataset_builder = tfds.builder(tfds_name, data_dir=data_dir)
        dataset_split = deterministic_data.get_read_instruction_for_host(
            split,
            dataset_builder.info.splits[split].num_examples,
        )
        dataset = deterministic_data.create_dataset(
            dataset_builder,
            split=dataset_split,
            batch_dims=(),
            num_epochs=1,
            shuffle=False,
        )

        lmdb_file = str(save_dir / split)
        env = lmdb.open(
            lmdb_file, map_size=1024**4, subdir=False, readonly=False, meminit=False
        )

        keys = []
        txn = env.begin(write=True)
        t0 = time.time()
        for i, sample in enumerate(dataset):
            if i % write_freq == 0:
                print(f"{i:06d}")
                txn.commit()
                txn = env.begin(write=True)
            sample2 = _convert_nested_mapping(sample)
            sample_key = f"{i:06d}".encode("ascii")
            keys.append(sample_key)
            txn.put(sample_key, compress(sample2))

        txn.commit()
        print((time.time() - t0) / (i + 1))  # 0.0298842241987586

        txn = env.begin(write=True)
        txn.put(b"__keys__", compress(keys))
        txn.commit()
        env.close()


def main_lmdb():
    """Wrap the LMDB file reading into PyTorch Dataset object,
    which is much faster than the original TFRecord version.
    As compressed 5x, the dataset can be put in RAM disk /dev/shm/ for extra speed.
    """
    TRANSFORM_TRAIN = (
        VideoFromTfds(),
        SparseToDenseAnnotation(max_instances=23),
        TemporalRandomStridedWindow(length=6),  # 24|6
        ResizeSmall(size=64),  # 128|64
        FlowToRgb(),
    )
    TRANSFORM_EVAL = (
        VideoFromTfds(),
        SparseToDenseAnnotation(max_instances=23),
        TemporalCropOrPad(length=24),
        ResizeSmall(size=64),  # 128|64
        FlowToRgb(),
    )
    batch_size = 16
    num_worker = min(16, batch_size)  # os.cpu_count()
    movi = MoviLmdb(
        "/dev/shm/movi_a/train", transform=TRANSFORM_TRAIN, max_spare=num_worker
    )
    dataloader = ptud.DataLoader(
        movi,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        collate_fn=MoviLmdb.collate_fn,
    )
    t0 = time.time()
    for i, batch in enumerate(dataloader):
        print("#" * 10, i, {k: list(v.shape) for k, v in batch.items()})
    print((time.time() - t0) / (i + 1))  # 0.029463951827273925


if __name__ == "__main__":
    main_convert_tfds_to_lmdb()
    # main_lmdb()

"""
'instances', 
    'quaternions', 'angular_velocities', 'material_label', 'friction', 'size_label', 'velocities', 'restitution', 'color', 'mass', 'color_label', 'bboxes', 'image_positions', 'bbox_frames', 'positions', 'shape_label', 'visibility', 'bboxes_3d'
'events', 
    collisions
        'position', 'instances', 'force', 'frame', 'image_position', 'contact_normal'
'metadata', 
    'height', 'forward_flow_range', 'backward_flow_range', 'width', 'depth_range', 'num_frames', 'num_instances', 'video_name'
'backward_flow', 
'camera', 
    'quaternions', 'field_of_view', 'positions', 'focal_length', 'sensor_width'
'normal', 
'object_coordinates', 
'depth', 
'segmentations', 
'video', 
'forward_flow'
"""
