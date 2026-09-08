import torch

ckpt_path = "randla_net/output/checkpoint.tar"
checkpoint = torch.load(ckpt_path, map_location="cpu")

print("Top-level keys in checkpoint:")
print(list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint))
print()

state_dict = None
if isinstance(checkpoint, dict):
    for key in ["state_dict", "model_state_dict", "model", "net"]:
        if key in checkpoint:
            state_dict = checkpoint[key]
            print(f"Found weights under key: '{key}'")
            break
    if state_dict is None:
        state_dict = checkpoint

print()
print(f"Number of layers/params in state_dict: {len(state_dict)}")
print()
print("First 5 layer names and shapes:")
for i, (name, tensor) in enumerate(state_dict.items()):
    print(f"  {name}: {tuple(tensor.shape)}")
    if i >= 4:
        break

print()
print("Last 5 layer names and shapes:")
items = list(state_dict.items())
for name, tensor in items[-5:]:
    print(f"  {name}: {tuple(tensor.shape)}")