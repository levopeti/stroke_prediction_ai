import torch
from training.utils.lit_model import LitModel

# "./models/2024-01-28-12-37_non-inverted_90_arm/epoch=79-step=8160.ckpt"
# "./models/2024-01-30-17-13_inverted_90_arm/epoch=28-step=2958.ckpt"
# "./models/2024-01-31-07-12_non-inverted_60_arm/epoch=48-step=4998.ckpt"
# "./models/2024-02-01-01-41_inverted_60_arm/epoch=0-step=102.ckpt"
# "./models/2024-02-01-08-51_non-inverted_30_arm/epoch=60-step=6222.ckpt"
# "./models/2024-02-02-06-41_inverted_30_arm/epoch=2-step=306.ckpt"
ckpt_path = "./models/2024-02-02-06-41_inverted_30_arm/epoch=2-step=306.ckpt"
lit_model = LitModel.load_from_checkpoint(ckpt_path)

model_scripted = torch.jit.script(lit_model.model)  # Export to TorchScript
model_scripted.save("./models/inverted_30_arm_2024-02-02-06-41.pt")  # 2024-01-28-12-37.pt

