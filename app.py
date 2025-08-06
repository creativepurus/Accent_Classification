# ------------------- Type "python app.py" in TERMINAL to Run the App -------------------

import torch
import torchaudio
import gradio as gr
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from safetensors.torch import load_file
import torch.nn as nn
import torch.nn.functional as F

# ------------------- Label Mapping -------------------

id2label = {
    0: "Canadian English",
    1: "England English"
}

# ------------------- Load Processor -------------------

processor = Wav2Vec2Processor.from_pretrained("Model")

# ------------------- Define Model -------------------

class Wav2Vec2Classifier(nn.Module):
    def __init__(self, num_labels):
        super(Wav2Vec2Classifier, self).__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.wav2vec2.config.hidden_size, num_labels)

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs.last_hidden_state
        pooled_output = hidden_states.mean(dim=1)
        logits = self.classifier(self.dropout(pooled_output))
        return logits

# ------------------- Load Weights -------------------

model = Wav2Vec2Classifier(num_labels=2)
state_dict = load_file("Model/checkpoint-276/model.safetensors", device="cpu")
model.load_state_dict(state_dict)
model.eval()

# ------------------- Prediction Function -------------------

def predict(audio_path):
    # Load & preprocess audio
    speech_array, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        speech_array = resampler(speech_array)

    inputs = processor(
        speech_array.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=16000 * 4
    )

    with torch.no_grad():
        logits = model(inputs.input_values)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()

    return id2label[pred_id]

# ------------------- Gradio UI with Dark Theme -------------------

with gr.Blocks(
    theme=gr.themes.Monochrome(primary_hue="blue", secondary_hue="purple", neutral_hue="slate"),
    css="""
        body { background-color: #1E1E2F !important; color: #E0E0E0 !important; }
        .gr-button { background-color: #3B82F6 !important; color: white !important; font-weight: bold; }
        .gr-textbox { font-size: 18px; }
        .gr-audio label { color: white !important; }
    """
) as demo:
    gr.Markdown(
        """
        <h1 style="text-align: center; color: #00FFFF;">🌍 Accent Classifier using Wav2Vec2</h1>
        <p style="text-align: center; font-size: 16px;">Upload or record a 4-second <b>English voice clip</b><br>
        This AI model detects whether your accent is <span style='color: #3B82F6; font-weight: bold;'>Canadian</span> or <span style='color: #FF4C4C; font-weight: bold;'>British</span>.</p>
        <br>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(type="filepath", label="🎧 Upload or Record English Voice")
            submit_btn = gr.Button("🔍 Detect Accent")

        with gr.Column(scale=1):
            label_output = gr.Text(label="🗣️ Predicted Accent")

    submit_btn.click(fn=predict, inputs=audio_input, outputs=label_output)

    gr.Markdown("---")
    gr.Markdown(
        "<p style='text-align: center;'>👨‍💻 Created by <a href='https://github.com/creativepurus' target='_blank' style='color:#66CFFF;'>Anand Purushottam</a> | <a href='https://www.linkedin.com/in/creativepurus/' target='_blank' style='color:#66CFFF;'>LinkedIn</a></p>"
    )

if __name__ == "__main__":
    demo.launch()