from qwen_vl_utils import process_vision_info

def qwenvl_forward(model, processor, prompt, img_paths, temperature=0.9):
    """
    Run local Qwen-VL inference with multi-image + text chat format.

    Inputs:
    - model: loaded Qwen-VL model
    - processor: corresponding processor/tokenizer
    - prompt: user text prompt
    - img_paths: list of image file paths
    - temperature: generation temperature

    Returns:
    - decoded model response text
    """
    # ==========================================================
    # [A] Build chat-style multimodal message payload.
    # ==========================================================
    messages = [
        {
            'role': "system",
            "content": 'You are a highly professional and experienced clinician. You are familiar with the latest medical guidelines, diagnostic standards and treatment plans, and can reasonably answer users\' questions.'
        },
        {
            "role": "user",
            "content": [],
        },
    ]
    
    # Attach all images to the user message in order.
    for img_path in img_paths:
        messages[1]["content"].append({
            "type": "image",
            "image": img_path,
        })
    
    # Append the text prompt after image items.
    messages[1]["content"].append({"type": "text", "text": prompt})

    # ==========================================================
    # [B] Convert messages to model-ready tensors.
    # ==========================================================
    # Chat template serializes role/content into one generation prompt string.
    texts = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Extract processed vision inputs expected by the Qwen VL processor.
    image_inputs, video_inputs = process_vision_info(messages)  

    # inputs have input_ids + vision tensors, all moved to the model's device in the processor call.
    # vision tensors are already preprocessed and formatted by process_vision_info to match the model's expected input format for images/videos.
    inputs = processor(
        text=[texts],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    # ==========================================================
    # [C] Generate, trim prompt tokens, and decode final text.
    # ==========================================================
    # **inputs contains both tokenized text and processed vision tensors, all on the correct device.
    generated_ids = model.generate(**inputs, max_new_tokens=2048, repetition_penalty=1, temperature=temperature)
    
    # Remove input prompt tokens so decoding contains only newly generated output.
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    # Decode the trimmed generated token IDs back into text.
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0] # e.g. if batch size is 1, we take the first element of the output list.

    return output_texts
