print("hehe")

from flask import Flask, request, jsonify

from transformers import AutoTokenizer

import torch

from modeling import create_model
from scaler import Scaler, load_pretrained_scaler
from inference_api import transform_input, predict_price

print("hihi")

config_file = "t5config.json"
pretrained_base_model_name = "VietAI/vit5-base"

app = Flask(__name__)

use_gpu = False
# Khởi tạo mô hình
device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
model = create_model(config_file, state_dict_path="checkpoint/model_state_dict.pt", device=device)
model.eval()

# Khởi tạo tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_base_model_name)

# Load Scaler
scaler = load_pretrained_scaler("scaler.pt")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(data)
    # data: {"estate_type", "square", "description"}

    # Xử lý dữ liệu đầu vào
    input_encoding = transform_input(tokenizer, data)

    # Dự đoán giá
    price = predict_price(model, scaler, input_encoding)

    return jsonify({'price': price})

if __name__ == '__main__':
    app.run(port=5000)