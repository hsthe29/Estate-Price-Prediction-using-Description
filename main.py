from flask import Flask, render_template

from flask import Flask, render_template
from transformers import AutoTokenizer

from form import InputForm
from inference_api import transform_input, predict_price
from modeling import load_pretrained_model
from scaler import Scaler, load_pretrained_scaler

app = Flask(__name__)

app.config['SECRET_KEY'] = '7d0fb494ff22bc777a2384efae5670f5b6c41fafb4738df9cd4f50741d15dd78'
DEFAULT_PRICE = "0.0 VND"
CONFIG_FILE = "t5config.json"
PRETRAINED_BASE_MODEL_NAME = "VietAI/vit5-base"


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = InputForm()
    predicted_price = DEFAULT_PRICE
    if form.validate_on_submit():
        input_encoding = transform_input(tokenizer, form.data)
        price = predict_price(model, scaler, input_encoding)
        predicted_price = f"{price:,}" + " VND"

    return render_template('home.html', title='Home', form=form, predicted_price=predicted_price)


if __name__ == '__main__':
    # Khởi tạo mô hình
    model = load_pretrained_model(CONFIG_FILE,
                                  state_dict_path="checkpoint/model_state_dict.pt",
                                  use_gpu=False,
                                  train=False)
    
    # Khởi tạo tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_BASE_MODEL_NAME)
    
    # Load Scaler
    scaler = load_pretrained_scaler("scaler.pt")
    
    app.run(host='localhost', port=3000, debug=True)
