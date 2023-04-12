import simpleaudio as sa

# Load the WAV file
wave_obj = sa.WaveObject.from_wave_file('aaa.wav')

# Play the WAV file
play_obj = wave_obj.play()
play_obj.wait_done()