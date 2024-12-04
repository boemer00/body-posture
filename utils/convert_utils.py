from pydub import AudioSegment
import os

def convert_mp3_to_wav(mp3_file, output_dir=None):
    """
    Converts an MP3 file to a WAV file.

    Args:
        mp3_file (str): Path to the MP3 file.
        output_dir (str, optional): Directory to save the WAV file. Defaults to the same directory as the MP3 file.

    Returns:
        str: Path to the converted WAV file.
    """
    if not os.path.exists(mp3_file):
        raise FileNotFoundError(f"The file {mp3_file} does not exist.")

    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(mp3_file)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate output WAV file path
    base_name = os.path.splitext(os.path.basename(mp3_file))[0]
    wav_file = os.path.join(output_dir, f"{base_name}.wav")

    # Convert MP3 to WAV
    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(wav_file, format="wav")
    print(f"Converted {mp3_file} to {wav_file}")

    return wav_file

# Example usage
if __name__ == "__main__":
    mp3_path = "/Users/renatoboemer/code/developer/body-posture/app/alert.mp3"
    wav_path = convert_mp3_to_wav(mp3_path)
    print(f"Output WAV file: {wav_path}")
