(bart) kaguya@kaguyaserver:~/usth/led-clinical$ python scripts/train.py --config configs/train_config.yaml
2026-04-16 16:25:40,007 [INFO] Loaded config from configs/train_config.yaml
2026-04-16 16:25:40,277 [INFO] HTTP Request: HEAD https://huggingface.co/allenai/led-base-16384/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-04-16 16:25:40,282 [INFO] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/allenai/led-base-16384/38335783885b338d93791936c54bb4be46bebed9/config.json "HTTP/1.1 200 OK"
2026-04-16 16:25:40,538 [INFO] HTTP Request: HEAD https://huggingface.co/allenai/led-base-16384/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
2026-04-16 16:25:40,545 [INFO] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/allenai/led-base-16384/38335783885b338d93791936c54bb4be46bebed9/tokenizer_config.json "HTTP/1.1 200 OK"
2026-04-16 16:25:40,804 [INFO] HTTP Request: GET https://huggingface.co/api/models/allenai/led-base-16384/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
2026-04-16 16:25:41,058 [INFO] HTTP Request: GET https://huggingface.co/api/models/allenai/led-base-16384/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
2026-04-16 16:25:41,356 [INFO] HTTP Request: HEAD https://huggingface.co/allenai/led-base-16384/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-04-16 16:25:41,363 [INFO] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/allenai/led-base-16384/38335783885b338d93791936c54bb4be46bebed9/config.json "HTTP/1.1 200 OK"
2026-04-16 16:25:41,615 [INFO] HTTP Request: HEAD https://huggingface.co/allenai/led-base-16384/resolve/main/model.safetensors "HTTP/1.1 404 Not Found"
2026-04-16 16:25:41,877 [INFO] HTTP Request: HEAD https://huggingface.co/allenai/led-base-16384/resolve/main/model.safetensors.index.json "HTTP/1.1 404 Not Found"
2026-04-16 16:25:42,161 [INFO] HTTP Request: HEAD https://huggingface.co/allenai/led-base-16384/resolve/main/pytorch_model.bin "HTTP/1.1 302 Found"
2026-04-16 16:25:42,407 [INFO] HTTP Request: HEAD https://huggingface.co/allenai/led-base-16384/resolve/main/model.safetensors "HTTP/1.1 404 Not Found"
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 299/299 [00:00<00:00, 54768.84it/s]
2026-04-16 16:25:42,654 [INFO] HTTP Request: GET https://huggingface.co/api/models/allenai/led-base-16384 "HTTP/1.1 200 OK"
2026-04-16 16:25:42,783 [INFO] HTTP Request: HEAD https://huggingface.co/allenai/led-base-16384/resolve/main/generation_config.json "HTTP/1.1 307 Temporary Redirect"
2026-04-16 16:25:42,789 [INFO] HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/allenai/led-base-16384/38335783885b338d93791936c54bb4be46bebed9/generation_config.json "HTTP/1.1 200 OK"
[Model] Vocab size OK — 50265 tokens
[Model] Loaded allenai/led-base-16384
  Total params:     161,844,480
  Trainable params: 161,844,480
2026-04-16 16:25:42,939 [INFO] HTTP Request: GET https://huggingface.co/api/models/allenai/led-base-16384/commits/main "HTTP/1.1 200 OK"
2026-04-16 16:25:43,224 [INFO] HTTP Request: GET https://huggingface.co/api/models/allenai/led-base-16384/discussions?p=0 "HTTP/1.1 200 OK"
2026-04-16 16:25:43,523 [INFO] HTTP Request: GET https://huggingface.co/api/models/allenai/led-base-16384/commits/refs%2Fpr%2F4 "HTTP/1.1 200 OK"
[DrugDictionary] Loaded 172616 drug entries from 4 source categories
[DrugDictionary] Aho-Corasick automaton built (172616 patterns)
2026-04-16 16:25:43,915 [INFO] HTTP Request: HEAD https://huggingface.co/allenai/led-base-16384/resolve/refs%2Fpr%2F4/model.safetensors.index.json "HTTP/1.1 404 Not Found"
2026-04-16 16:25:43,916 [INFO] Pre-baked corruption detected — skipping DrugDictionary load.
2026-04-16 16:25:44,169 [INFO] HTTP Request: HEAD https://huggingface.co/allenai/led-base-16384/resolve/refs%2Fpr%2F4/model.safetensors "HTTP/1.1 302 Found"
[Dataset] Loaded 212038 records from data/processed/train.jsonl (train=True, corruption_rate=0.3)
[Dataset] Loaded 26503 records from data/processed/val.jsonl (train=False, corruption_rate=0.0)
2026-04-16 16:25:50,353 [INFO] Total steps: 132,520  |  Warmup steps: 13,252 (10% of 132,520)
2026-04-16 16:25:50,617 [INFO] Starting training...
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'eos_token_id': 2, 'bos_token_id': 0, 'pad_token_id': 1}.
  0%|                                                                                                                                                                                                              | 0/132530 [00:00<?, ?it/s]/pytorch/aten/src/ATen/native/cuda/IndexKernelUtils.cu:16: vectorized_gather_kernel: block: [1026,0,0], thread: [96,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernelUtils.cu:16: vectorized_gather_kernel: block: [1026,0,0], thread: [97,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernelUtils.cu:16: vectorized_gather_kernel: block: [1026,0,0], thread: [98,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernelUtils.cu:16: vectorized_gather_kernel: block: [1026,0,0], thread: [99,0,0] Assertion `ind >=0 && ind < ind_dim_size && "vectorized gather kernel index out of bounds"` failed.

Traceback (most recent call last):
  File "/home/kaguya/usth/led-clinical/scripts/train.py", line 217, in <module>
    main()
  File "/home/kaguya/usth/led-clinical/scripts/train.py", line 207, in main
    trainer.train(resume_from_checkpoint=resume_from)
  File "/home/kaguya/miniforge3/envs/bart/lib/python3.11/site-packages/transformers/trainer.py", line 1425, in train
    return inner_training_loop(
           ^^^^^^^^^^^^^^^^^^^^
  File "/home/kaguya/miniforge3/envs/bart/lib/python3.11/site-packages/transformers/trainer.py", line 1507, in _inner_training_loop
    self._run_epoch(
  File "/home/kaguya/miniforge3/envs/bart/lib/python3.11/site-packages/transformers/trainer.py", line 1735, in _run_epoch
    tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/kaguya/miniforge3/envs/bart/lib/python3.11/site-packages/transformers/trainer.py", line 1907, in training_step
    loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/kaguya/miniforge3/envs/bart/lib/python3.11/site-packages/transformers/trainer.py", line 1979, in compute_loss
    outputs = model(**inputs)
              ^^^^^^^^^^^^^^^
  File "/home/kaguya/miniforge3/envs/bart/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/kaguya/miniforge3/envs/bart/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/kaguya/miniforge3/envs/bart/lib/python3.11/site-packages/accelerate/utils/operations.py", line 823, in forward
    return model_forward(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/kaguya/miniforge3/envs/bart/lib/python3.11/site-packages/accelerate/utils/operations.py", line 811, in __call__
    return convert_to_fp32(self.model_forward(*args, **kwargs))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/kaguya/miniforge3/envs/bart/lib/python3.11/site-packages/torch/amp/autocast_mode.py", line 44, in decorate_autocast
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/kaguya/miniforge3/envs/bart/lib/python3.11/site-packages/transformers/models/led/modeling_led.py", line 2047, in forward
    masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/kaguya/miniforge3/envs/bart/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/kaguya/miniforge3/envs/bart/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/kaguya/miniforge3/envs/bart/lib/python3.11/site-packages/torch/nn/modules/loss.py", line 1394, in forward
    return F.cross_entropy(
           ^^^^^^^^^^^^^^^^
  File "/home/kaguya/miniforge3/envs/bart/lib/python3.11/site-packages/torch/nn/functional.py", line 3504, in cross_entropy
    return torch._C._nn.cross_entropy_loss(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

  0%|                                                                                                                                                                                                              | 0/132530 [00:02<?, ?it/s]
(bart) kaguya@kaguyaserver:~/usth/led-clinical$ 




Lỗi này rất rõ ràng: `vectorized_gather_kernel` index out-of-bounds trong cross-entropy loss, tức là **labels chứa token ID nằm ngoài khoảng `[0, vocab_size)` hoặc không phải `-100`**.

Vocab size là 50265, nhưng một số label IDs đang vượt quá giới hạn đó.

**Root cause có thể:**

1. **Data corruption pipeline** — dataset load với `corruption_rate=0.3`, và log có dòng `"Pre-baked corruption detected"`. Nếu corruption logic sinh ra token IDs > 50265 hoặc giá trị âm khác `-100`, đây chính là thủ phạm.

2. **Tokenizer/model vocab mismatch** — log cảnh báo:
   > *"The tokenizer has new PAD/BOS/EOS tokens that differ from the model config"*
   
   Nếu tokenizer được add thêm special tokens nhưng model embedding chưa được resize, một số IDs sẽ out-of-range.

---

**Cách debug nhanh nhất** — thêm một đoạn check vào training loop hoặc dataset collator:

```python
# Trong collate_fn hoặc trước khi feed vào model
labels = batch["labels"]
vocab_size = model.config.vocab_size  # 50265

invalid_mask = (labels != -100) & ((labels < 0) | (labels >= vocab_size))
if invalid_mask.any():
    bad_ids = labels[invalid_mask].unique().tolist()
    print(f"[ERROR] Invalid label IDs found: {bad_ids}")
    # Clamp chúng về -100 để ignore
    labels[invalid_mask] = -100
```

---

**Fix dứt điểm theo từng nguyên nhân:**

**Nếu do corruption pipeline:**

Tìm chỗ corrupt labels trong dataset và clamp lại:
```python
# Trong __getitem__ của Dataset, sau khi tạo labels
labels = torch.clamp(labels, min=-100, max=self.tokenizer.vocab_size - 1)
# Hoặc đặt invalid thành -100 (ignore index)
labels[(labels >= self.tokenizer.vocab_size) | ((labels < 0) & (labels != -100))] = -100
```

**Nếu do vocab mismatch (add special tokens):**

Sau khi add tokens vào tokenizer, phải resize model:
```python
tokenizer.add_special_tokens({...})
model.resize_token_embeddings(len(tokenizer))
```

---

**Quick sanity check** — chạy lệnh này để xem phân bố label IDs trong data trước khi train:

```python
import json, numpy as np

bad = 0
for line in open("data/processed/train.jsonl"):
    rec = json.loads(line)
    ids = rec.get("labels", [])
    for x in ids:
        if x != -100 and (x < 0 or x >= 50265):
            bad += 1

print(f"Invalid label tokens: {bad}")
```

Chạy cái đó trước, kết quả sẽ xác nhận ngay nguyên nhân là data hay model.