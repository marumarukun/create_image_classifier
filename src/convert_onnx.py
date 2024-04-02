
import onnx
import torch
from onnxsim import simplify

from model import MobileNetV3Trainer


# 最良のモデルを読み込む
best_model_path = './checkpoints/best-checkpoint-v2.ckpt'
model = MobileNetV3Trainer.load_from_checkpoint(checkpoint_path=best_model_path, max_epochs=10, num_classes=2)

model.cpu()

# モデルを評価モードに設定
model.eval()

# ダミー入力を作成
dummy_input = torch.randn(1, 3, 224, 224)

# ONNX形式に変換
torch.onnx.export(
    model,
    dummy_input,
    "onnx/mobilenetv3.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    dynamic_axes={"input": {0: "batch_size"}},
)

# 型の推定
model = onnx.load("onnx/mobilenetv3.onnx")
model = onnx.shape_inference.infer_shapes(model)

# モデル構造の最適化
model_simp, check = simplify(model)

onnx.save(model_simp, "onnx/mobilenetv3.onnx")

print("Model converted to ONNX format.")
