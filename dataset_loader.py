import os
import collections
import random
import tensorflow.compat.v1 as tf
import util


def data_loader(
        data_path,
        epochs,
        batch_size,
        training=True,
        load_estimated_rot=False):
    """Load stereo image datasets.

    Args:
      data_path: (string)
      epochs: (int) the number of training epochs.
      batch_size: (int) batch size.
      training: (bool) set it True when training to enable illumination randomization
          for input images.
      load_estimated_rot: (bool) set it True when training DirectionNet-T to load
          estimated rotation from DirectionNet-R saved as 'rotation_pred' on disk.

    Returns:
      Tensorflow Dataset
    """

    def load_data(path):
        """Load files saved as pickle."""
        img_id, rotation = tf.py_func(util.read_pickle,
                                      [path + '/rotation_gt.pickle'], [tf.string, tf.float32])
        _, translation = tf.py_func(util.read_pickle,
                                    [path + '/epipoles_gt.pickle'], [tf.string, tf.float32])
        _, fov = tf.py_func(util.read_pickle,
                            [path + '/fov.pickle'], [tf.string, tf.float32])

        if load_estimated_rot:
            _, rotation_pred = tf.py_func(util.read_pickle,
                                          [path + '/rotation_pred.pickle'], [tf.string, tf.float32])
        else:
            rotation_pred = tf.zeros_like(rotation)

        img_path = path + '/' + img_id
        return tf.data.Dataset.from_tensor_slices(
            (img_id, img_path, rotation, translation, fov, rotation_pred))

    def load_images(img_id, img_path, rotation, translation, fov, rotation_pred):
        """Load images and decode text lines."""

        def load_single_image(img_path):
            image = tf.image.decode_png(tf.read_file(img_path))
            image = tf.image.convert_image_dtype(image, tf.float32)
            image.set_shape([512, 512, 3])
            image = tf.squeeze(
                tf.image.resize(tf.expand_dims(image, 0), [256, 256],
                                method=tf.image.ResizeMethod.AREA))
            return image

        input_pair = collections.namedtuple(
            'data_input',
            [
                'id',
                'src_image',
                'trt_image',
                'rotation',
                'translation',
                'fov',
                'rotation_pred'
            ])

        src_image = load_single_image(img_path + '.src.perspective.png')
        trt_image = load_single_image(img_path + '.trt.perspective.png')

        random_gamma = random.uniform(0.7, 1.2)
        if training:
            src_image = tf.image.adjust_gamma(src_image, random_gamma)
            trt_image = tf.image.adjust_gamma(trt_image, random_gamma)

        #rotation = tf.reshape(
        #    tf.stack([tf.decode_csv(rotation, [0.0] * 9)], 0), [3, 3])
        #rotation.set_shape([3, 3])

        #translation = tf.reshape(
        #    tf.stack([tf.decode_csv(translation, [0.0] * 3)], 0), [3])
        #translation.set_shape([3])

        #fov = tf.reshape(tf.stack([tf.decode_csv(fov, [0.0])], 0), [1])
        #fov.set_shape([1])
        
        # In TF2 eager mode, rotation/translation/fov are already float tensors
        rotation = tf.reshape(rotation, [3, 3])
        translation = tf.reshape(translation, [3])
        fov = tf.reshape(fov, [1])

        # Optional: explicitly set shapes for clarity
        rotation.set_shape([3, 3])
        translation.set_shape([3])
        fov.set_shape([1])


        if load_estimated_rot:
            rotation_pred = tf.reshape(
                tf.stack([tf.decode_csv(rotation_pred, [0.0] * 9)], 0), [3, 3])
            rotation_pred.set_shape([3, 3])

        return input_pair(img_id, src_image, trt_image, rotation, translation, fov, rotation_pred)

    # 1) Dosya glob’unu açıkça topla (log için de kullanacağız)
    file_pattern = os.path.join(data_path, '*')
    files = tf.gfile.Glob(file_pattern)
    # Gizli dosyaları (.DS_Store gibi) ve klasör olmayan öğeleri filtrele
    files = [p for p in files if os.path.isdir(p) and not p.endswith('.DS_Store')]
    tf.logging.info("DATA GLOB -> %s  |  %d path gefunden", file_pattern, len(files))

    ds = tf.data.Dataset.from_tensor_slices(files)

    # 2) load_data: her path bir örnek dizini olmalı
    ds = ds.flat_map(load_data)

    # 3) HATALARI GEÇİCİ OLARAK YUTMA! (debug için kaldır)
    # .apply(tf.data.experimental.ignore_errors())

    # 4) Paralelizmi düşük tut (macOS/py_func için stabil)
    ds = ds.map(load_images, num_parallel_calls=1)

    # 5) drop_remainder=False, aksi halde küçük split'te 0 batch olabilir
    ds = ds.batch(batch_size, drop_remainder=False)

    # 6) Prefetch’i düşük tut; ilk batch’i çabuk görmek için
    ds = ds.prefetch(1)

    # 7) DEBUG: İlk örneği gerçekten üretiyor muyuz?
    # (Graf derlenirken yazdırır; ilk run’da görünür)
    debug_card = tf.data.experimental.cardinality(ds)
    tf.logging.info("DATASET CARDINALITY (bilirsek): %s", debug_card)

    return ds

