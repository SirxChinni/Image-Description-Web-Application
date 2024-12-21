from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # for allowing requests from another domain
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import io

# -------------------------------------------------
# 1. Initialize FastAPI
# -------------------------------------------------
app = FastAPI()

# (Optional) Configure CORS if you're serving frontend from a different origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # adjust for specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# 2. Load Model and Tokenizer at Startup
# -------------------------------------------------
model_name = "vit-gpt2-image-captioning"

# Download/Load the pretrained model, processor, and tokenizer
model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define generation parameters
max_length = 16
num_beams = 4
gen_kwargs = {
    "max_length": max_length,
    "num_beams": num_beams
}

# -------------------------------------------------
# 3. Inference Function
# -------------------------------------------------
def predict_step(image: Image.Image) -> str:
    """
    Takes a PIL Image, runs it through the model, and returns the generated caption.
    """
    # Preprocess image
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Generate output
    output_ids = model.generate(pixel_values, **gen_kwargs)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    caption = caption.strip()
    return caption

# -------------------------------------------------
# 4. Routes
# -------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Welcome to the Image Captioning API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives an image file (via multipart/form-data), 
    runs inference, and returns the image caption.
    """
    # Read file bytes
    file_bytes = await file.read()
    
    # Convert bytes to a PIL image
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    # Run inference
    caption = predict_step(image)
    
    # Return JSON response
    return JSONResponse(content={"caption": caption})

# -------------------------------------------------
# 5. Run App (development)
# -------------------------------------------------
# To run:  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# Or simply: python -m uvicorn app:app --reload
