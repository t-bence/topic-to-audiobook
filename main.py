import sounddevice as sd
import torch
from transformers import AutoTokenizer, VitsModel


def main(topic: str):

    model = VitsModel.from_pretrained("facebook/mms-tts-hun")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-hun")

    inputs = tokenizer(topic, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs).waveform

    # Squeeze the waveform to remove the batch dimension if it's there (e.g., [1, N] -> [N])
    # and convert to a NumPy array for sounddevice
    waveform_np = output.squeeze().cpu().numpy()
    sampling_rate = model.config.sampling_rate

    print("Playing...")
    sd.play(waveform_np, samplerate=sampling_rate)
    sd.wait()
    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate speech from text.")
    parser.add_argument(
        "text",
        type=str,
        nargs="?",  # Add this to make the positional argument optional
        help="The text to speak.",
        default="Boribon és Annipanni elmentek szőlőt szedni.",
    )
    args = parser.parse_args()

    main(args.text)
