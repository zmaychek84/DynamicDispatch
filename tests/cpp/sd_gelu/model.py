import os
import numpy as np
import onnx
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_opsetid,
    make_tensor_value_info,
    make_tensor,
)
from onnx.checker import check_model
import onnxruntime


def shape2tuple(shape):
    return tuple(getattr(d, "dim_value", 0) for d in shape.dim)

def create_gelu_model(B, M, N, InT, OutT):
    X = make_tensor_value_info("X", InT, [B, M, N])
    Y = make_tensor_value_info("Y", OutT, [B, M, N])

    wts = np.zeros(64, dtype=np.uint16) # 128 bytes dummy wts
    wts_tsor = make_tensor("w", onnx.TensorProto.UINT16, wts.shape, wts)

    gelu = make_node(
        name="gelu",
        op_type="SDGelu",
        inputs=["X", "w"],
        outputs=["Y"],
        domain="com.amd",
    )

    graph = make_graph(
        [gelu], "lr", [X], [Y], initializer=[wts_tsor]
    )

    graph.node[0].attribute.append(onnx.helper.make_attribute("input_shape", [B, M, N]))
    graph.node[0].attribute.append(onnx.helper.make_attribute("in_dtypes", ["bfloat16", "bfloat16"]))
    graph.node[0].attribute.append(onnx.helper.make_attribute("out_dtypes", ["bfloat16"]))
    graph.node[0].attribute.append(onnx.helper.make_attribute("output_shape", [B, M, N]))

    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 19)])
    return onnx_model


if __name__ == "__main__":
    B, M, N = (2, 4096, 1280)
    dir_name = "test_sdgelu"
    os.makedirs(dir_name, exist_ok=True)

    onnx_model = create_gelu_model(
        B, M, N, onnx.TensorProto.BFLOAT16, onnx.TensorProto.BFLOAT16
    )
    onnx.save(onnx_model, f"{dir_name}/sdgelu.onnx")

    from ryzenai_dynamic_dispatch import onnx_graph as ogm
    from ryzenai_dynamic_dispatch import fuse

    metainfo = fuse.prepare_metadata(ogm.ONNXGraph(onnx_model), dir_name)
    json_str = fuse.save_tensors_to_json(
        f"{dir_name}/model_gelu_meta.json", *metainfo
    )
    print("JSON Metadata saved to", f"{dir_name}/model_sdgelu_meta.json")
