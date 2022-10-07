import os
import tensorflow as tf
import urllib.request

_LPIPS_URL = "http://rail.eecs.berkeley.edu/models/lpips/net-lin_alex_v0.1.pb"


def ensure_lpips_weights_exist(weight_path_out):
    """Downloads weights if needed."""
    if os.path.isfile(weight_path_out):
        return
    print("Downloading LPIPS weights:", _LPIPS_URL, "->", weight_path_out)
    urllib.request.urlretrieve(_LPIPS_URL, weight_path_out)
    if not os.path.isfile(weight_path_out):
        raise ValueError(f"Failed to download LPIPS weights from {_LPIPS_URL} "
                         f"to {weight_path_out}. Please manually download!")


class LPIPSLoss(object):
    """Calcualte LPIPS loss."""

    def __init__(self, weight_path):
        ensure_lpips_weights_exist(weight_path)

        def wrap_frozen_graph(graph_def, inputs, outputs):
            def _imports_graph_def():
                tf.graph_util.import_graph_def(graph_def, name="")

            wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
            import_graph = wrapped_import.graph
            return wrapped_import.prune(
                tf.nest.map_structure(import_graph.as_graph_element, inputs),
                tf.nest.map_structure(import_graph.as_graph_element, outputs))

        # Pack LPIPS network into a tf function
        graph_def = tf.compat.v1.GraphDef()
        with open(weight_path, "rb") as f:
            graph_def.ParseFromString(f.read())
        self._lpips_func = tf.function(
            wrap_frozen_graph(
                graph_def, inputs=("0:0", "1:0"), outputs="Reshape_10:0"))

    def __call__(self, fake_image, real_image):
        """Assuming inputs are in [0, 1]."""

        # Move inputs to [-1, 1] and NCHW format.
        def _transpose_to_nchw(x):
            return tf.transpose(x, (0, 3, 1, 2))

        fake_image = _transpose_to_nchw(fake_image * 2 - 1.0)
        real_image = _transpose_to_nchw(real_image * 2 - 1.0)
        loss = self._lpips_func(fake_image, real_image)
        return tf.reduce_mean(loss)  # Loss is N111, take mean to get scalar.