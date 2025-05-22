# Define model
from flask import Flask, request, render_template
from PIL import Image, ImageOps
import torch
import torch
from training import CNNModel,transform


# Later: load model
model = CNNModel()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if file:
        img = Image.open(file).convert('L')
        img = ImageOps.invert(img)
        img = ImageOps.autocontrast(img)
        img = ImageOps.pad(img, (20, 20), color=0)  # Padding to keep aspect ratio
        img = img.resize((28, 28), Image.Resampling.LANCZOS)

        img = transform(img).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = model(img)
            predicted = torch.argmax(output, 1).item()
        return f'Predicted digit: {predicted}'
    return 'No image uploaded'

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
