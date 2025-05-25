import asyncio
import os

import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer

load_dotenv()


assert os.environ["OPENAI_API_KEY"], "No API key found"


async def main(story: str, voice: str = "ash", instructions: str = "") -> None:
    openai = AsyncOpenAI()

    async with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=story,
        instructions=instructions,
        response_format="pcm",
    ) as response:
        await LocalAudioPlayer().play(response)


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    asyncio.run(
        main(
            config["story"], voice=config["voice"], instructions=config["instructions"]
        )
    )
