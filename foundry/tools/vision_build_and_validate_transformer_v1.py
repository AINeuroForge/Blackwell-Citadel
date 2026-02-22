import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

class TransformerLayer(nn.Module):
    def __init__(self, H=32, heads=4, mlp_mult=4):
        super().__init__()
        self.H = H
        self.heads = heads
        self.hd = H // heads
        Hff = H * mlp_mult

        self.q = nn.Linear(H, H)
        self.k = nn.Linear(H, H)
        self.v = nn.Linear(H, H)
        self.o = nn.Linear(H, H)

        self.ln1 = nn.LayerNorm(H)
        self.ln2 = nn.LayerNorm(H)

        self.fc1 = nn.Linear(H, Hff)
        self.fc2 = nn.Linear(Hff, H)

        self.gelu = nn.GELU()

    def split(self, x):
        B,S,H = x.shape
        x = x.view(B,S,self.heads,self.hd)
        return x.permute(0,2,1,3)

    def merge(self, x):
        B,heads,S,hd = x.shape
        x = x.permute(0,2,1,3)
        return x.reshape(B,S,self.H)

    def forward(self, x):
        q = self.split(self.q(x))
        k = self.split(self.k(x))
        v = self.split(self.v(x))

        scores = torch.matmul(q, k.transpose(-1,-2)) / (self.hd**0.5)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v)
        merged = self.merge(ctx)

        x = x + self.o(merged)
        x = self.ln1(x)

        h = self.fc1(x)
        h = self.gelu(h)
        h = self.fc2(h)

        x = x + h
        x = self.ln2(x)
        return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True)
    args = ap.parse_args()

    art = args.artifact_dir
    x = np.load(os.path.join(art, "x.npy"))

    model = TransformerLayer()
    model.eval()

    wpath = os.path.join(art, "weights.npz")
    if not os.path.exists(wpath):
        raise FileNotFoundError(f"Missing weights file: {wpath}")

    w = np.load(wpath)

    required = ["Wq","bq","Wk","bk","Wv","bv","Wo","bo","W1","b1","W2","b2","ln1_g","ln1_b","ln2_g","ln2_b"]
    missing = [k for k in required if k not in w.files]
    if missing:
        raise KeyError(f"weights.npz missing keys: {missing}. Found keys: {w.files}")

    with torch.no_grad():
        model.q.weight.copy_(torch.from_numpy(w["Wq"].T))
        model.q.bias.copy_(torch.from_numpy(w["bq"]))

        model.k.weight.copy_(torch.from_numpy(w["Wk"].T))
        model.k.bias.copy_(torch.from_numpy(w["bk"]))

        model.v.weight.copy_(torch.from_numpy(w["Wv"].T))
        model.v.bias.copy_(torch.from_numpy(w["bv"]))

        model.o.weight.copy_(torch.from_numpy(w["Wo"].T))
        model.o.bias.copy_(torch.from_numpy(w["bo"]))

        model.fc1.weight.copy_(torch.from_numpy(w["W1"].T))
        model.fc1.bias.copy_(torch.from_numpy(w["b1"]))

        model.fc2.weight.copy_(torch.from_numpy(w["W2"].T))
        model.fc2.bias.copy_(torch.from_numpy(w["b2"]))

        model.ln1.weight.copy_(torch.from_numpy(w["ln1_g"]))
        model.ln1.bias.copy_(torch.from_numpy(w["ln1_b"]))

        model.ln2.weight.copy_(torch.from_numpy(w["ln2_g"]))
        model.ln2.bias.copy_(torch.from_numpy(w["ln2_b"]))

    xt = torch.from_numpy(x).float()
    with torch.no_grad():
        yt = model(xt).numpy()

    onnx_path = os.path.join(art, "transformer_layer_v1.onnx")
    torch.onnx.export(
        model,
        xt,
        onnx_path,
        input_names=["x"],
        output_names=["y"],
        opset_version=19
    )

    sess = ort.InferenceSession(
        onnx_path,
        providers=["CUDAExecutionProvider","CPUExecutionProvider"]
    )

    y_onnx = sess.run(None, {"x": x.astype(np.float32)})[0]

    np.save(os.path.join(art, "y_onnx.npy"), y_onnx)
    print("Providers:", sess.get_providers())
    print("Vision OK")

if __name__ == "__main__":
    main()