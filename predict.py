"""
download the models to ./weights
wget https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt -P ./weights
wget https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt -P ./weights
wget https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt -P ./weights
wget https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt -P ./weights
wget https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large.pt -P ./weights
"""

import io
import os
from typing import Optional
import torch
from cog import BasePredictor, Input, Path, BaseModel

import whisper
from whisper.model import Whisper, ModelDimensions
from whisper.tokenizer import LANGUAGES
from whisper.utils import format_timestamp


class ModelOutput(BaseModel):
    detected_language: str
    transcription: str
    translation: Optional[str]
    txt_file: Optional[Path]
    srt_file: Optional[Path]


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        self.models = {}
        for model in ["tiny", "base", "small", "medium", "large"]:
            model_bytes = open(f"weights/{model}.pt", "rb").read()
            with io.BytesIO(model_bytes) as fp:
                checkpoint = torch.load(fp, map_location="cpu")

                dims = ModelDimensions(**checkpoint["dims"])
                state_dict = checkpoint["model_state_dict"]
                self.models[model] = Whisper(dims)
                self.models[model].load_state_dict(state_dict)

    def predict(
        self,
        audio: Path = Input(description="Audio file"),
        model: str = Input(
            default="base",
            choices=["tiny", "base", "small", "medium", "large"],
            description="Choose a Whisper model.",
        ),
        transcription: str = Input(
            choices=["plain text", "srt", "vtt"],
            default="plain text",
            description="Choose the format for the transcription",
        ),
        translate: bool = Input(
            default=False,
            description="Translate the text to English when set to True",
        ),
    ) -> ModelOutput:

        """Run a single prediction on the model"""
        print(f"Transcribe with {model} model")
        model = self.models[model].to("cuda")
        result = model.transcribe(str(audio))

        if transcription == "plain text":
            transcription = result["text"]
        elif transcription == "srt":
            transcription = write_srt(result["segments"])
        else:
            transcription = write_vtt(result["segments"])

        if translate:
            translation = model.transcribe(str(audio), task="translate")

        return ModelOutput(
            detected_language=LANGUAGES[result["language"]],
            transcription=transcription,
            translation=translation["text"] if translate else None,
        )


def write_vtt(transcript):
    result = "WEBVTT\n"
    for segment in transcript:
        result += f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n"
        result += f"{segment['text'].strip().replace('-->', '->')}\n"
    return result


def write_srt(transcript):
    result = ""
    for i, segment in enumerate(transcript, start=1):
        result += f"{i}\n"
        result += f"{format_timestamp(segment['start'], always_include_hours=True, decimal_marker=',')} --> "
        result += f"{format_timestamp(segment['end'], always_include_hours=True, decimal_marker=',')}\n"
        result += f"{segment['text'].strip().replace('-->', '->')}\n"
    return result
