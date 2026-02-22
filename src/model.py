import math
import torch
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    AutoModel,
    AutoConfig
)

def split_model(model_path):
    """
    Build a custom device_map for InternVL-style models on multi-GPU CUDA.

    Design assumption:
    - GPU 0 also hosts vision modules, so it is treated as "half capacity"
      for language layers.

    Returns:
    - device_map dict consumable by transformers.from_pretrained(..., device_map=...)
    """
    # ==========================================================
    # [A] Inspect model depth and available CUDA devices.
    # ==========================================================
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since GPU 0 also hosts vision modules, reduce its language-layer budget.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)

    # ==========================================================
    # [B] Assign language layers across GPUs.
    # ==========================================================
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1

    # ==========================================================
    # [C] Pin shared/vision/head modules to GPU 0.
    # ==========================================================
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def init_model(args):
    """
    Initialize local intern models based on CLI paths.

    Input:
    - args: parsed CLI namespace from run.py

    Returns:
    - model_set dict with shape:
      {
        "<model_name>": {
          "model": <loaded torch model>,
          "processor": <tokenizer/processor object>
        },
        ...
      }
    - Empty dict if nothing is successfully loaded.
    """
    # ==========================================================
    # [A] Output container for all loaded local models.
    # ==========================================================
    model_set = {}
    
    # Shared visual tokenization bounds used by Qwen VL processors.
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28

    # ==========================================================
    # [B] Load Qwen2.5-VL-7B intern model (if path is provided).
    # ==========================================================
    if args.qwen25_vl_7b_model_path:
        model_name = 'qwen25_vl_7b'
        print(f'Initializing {model_name}...')
        try:
            # device_map="auto" lets transformers shard modules automatically.
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.qwen25_vl_7b_model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2",
                trust_remote_code=True
            )
            # AutoProcessor handles text+image formatting for Qwen VL inputs.
            processor = AutoProcessor.from_pretrained(args.qwen25_vl_7b_model_path, min_pixels=min_pixels, max_pixels=max_pixels, trust_remote_code=True)
            model_set[model_name] = {'model': model, 'processor': processor}
            print(f'{model_name} loaded successfully.')
        except Exception as e:
            print(f"Error loading {model_name}: {e}")


    # ==========================================================
    # [C] Load Qwen2-VL-7B intern model (if path is provided).
    # ==========================================================
    if args.qwen2_vl_7b_model_path:
        model_name = 'qwen2_vl_7b'
        print(f'Initializing {model_name}...')
        try:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                args.qwen2_vl_7b_model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2",
                trust_remote_code=True
            )
            processor = AutoProcessor.from_pretrained(args.qwen2_vl_7b_model_path, min_pixels=min_pixels, max_pixels=max_pixels, trust_remote_code=True)
            model_set[model_name] = {'model': model, 'processor': processor}
            print(f'{model_name} loaded successfully.')
        except Exception as e:
            print(f"Error loading {model_name}: {e}")



    # ==========================================================
    # [D] Load InternVL3-8B intern model (if path is provided).
    # ==========================================================
    if args.internvl3_8b_model_path:
        model_name = 'internvl3_8b'
        print(f'Initializing {model_name}...')
        try:
            # Use custom split to balance language layers around vision load on GPU 0.
            device_map2 = split_model(args.internvl3_8b_model_path)
            model = AutoModel.from_pretrained(
                args.internvl3_8b_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                use_flash_attn=True, trust_remote_code=True, device_map=device_map2).eval()
            # InternVL path uses tokenizer-style processor object in this repo.
            processor = AutoTokenizer.from_pretrained(args.internvl3_8b_model_path, trust_remote_code=True)
            # Ensure batch padding uses EOS to avoid missing pad token issues.
            processor.pad_token_id = processor.eos_token_id
        
            model_set[model_name] = {'model': model, 'processor': processor}
            print(f'{model_name} loaded successfully.')
        except Exception as e:
            print(f"Error loading {model_name}: {e}")

    # ==========================================================
    # [E] Final sanity message.
    # ==========================================================
    if not model_set:
        print("Warning: No models were loaded. Check provided model paths in arguments.")

    return model_set
