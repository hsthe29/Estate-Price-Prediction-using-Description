import torch


def transform_input(tokenizer, input_dict):
    input_text = f"Loại: {input_dict['estate_type']}{tokenizer.eos_token}Diện tích: {float(input_dict['square'])}{tokenizer.eos_token}Mô tả: {input_dict['description']}"
    
    return tokenizer(input_text, return_tensors="pt")


def predict_price(model, scaler, input_encoding):
    with torch.no_grad():
        logits = model(**input_encoding)
    
    prices = scaler.invert(logits.cpu().numpy())
    return prices.item()
