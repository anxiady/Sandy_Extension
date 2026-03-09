import noisereduce as nr
import soundfile as sf


def clean_audio(input_file, output_file):
    data, rate = sf.read(input_file)
    reduced = nr.reduce_noise(
        y=data,
        sr=rate,
        stationary=True,
    )
    sf.write(output_file, reduced, rate)
