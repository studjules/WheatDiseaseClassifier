from flask import Flask, request, jsonify, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
from diseaseclassifier import Disease_detection

# Define the classes for classification
wheat_disease_classes = [
    'Aphid', 'Black Rust', 'Blast', 'Brown Rust', 'Common Root Rot',
    'Fusarium Head Blight', 'Healthy', 'Leaf Blight', 'Mildew', 'Mite',
    'Septoria', 'Smut', 'Stem fly', 'Tan spot', 'Yellow Rust'
]

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = Disease_detection()
checkpoint = torch.load("disease_classification_model.pth")
model.load_state_dict(checkpoint)
model.eval()

# Transformation for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/')
def home():
    return render_template('index.html')  # This will be your front-end

@app.route('/classify', methods=['POST'])
def classify_image():
    # Get the uploaded image
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file uploaded'})

    # Read and process the image
    image = Image.open(BytesIO(file.read()))
    image_tensor = transform(image).unsqueeze(0)

    # Predict the class
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
        predicted_class = wheat_disease_classes[predicted.item()]

    # Return the predicted class as a JSON response
    return jsonify({'predicted_class': predicted_class})

if __name__ == "__main__":
    app.run(debug=True)
