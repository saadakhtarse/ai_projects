import pyttsx3

def Speak(audio):
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.setProperty('rate', 150)
    print("")
    print(f"AI: {audio}")
    print("")
    engine.say(audio)
    engine.runAndWait()