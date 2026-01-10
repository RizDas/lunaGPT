import speech_recognition as sr
from colorama import Fore, init

init(autoreset=True)


def listen():
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = False
    recognizer.energy_threshold = 35000
    recognizer.dynamic_energy_adjustment_damping = 0.03
    recognizer.dynamic_energy_ratio = 1.9
    recognizer.pause_threshold = 0.4
    recognizer.operation_timeout = None
    recognizer.pause_threshold = 0.2
    recognizer.non_speaking_duration = 0.2

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

        print(Fore.LIGHTBLUE_EX + "I am Listening...", end="", flush=True)
        try:
            audio = recognizer.listen(source, timeout=None)
            print("\r" + Fore.LIGHTBLUE_EX + "Got it, Now Recognizing...", end="", flush=True)
            recognized_txt = recognizer.recognize_google(audio)
            print("\r" + " " * 50 + "\r", end="", flush=True)  # Clear the line
            return recognized_txt if recognized_txt else ""

        except sr.UnknownValueError:
            print("\r" + " " * 50 + "\r", end="", flush=True)  # Clear the line
            return ""
        except sr.RequestError as e:
            print("\r" + Fore.RED + f"Error with recognition service: {e}")
            return ""
        except Exception as e:
            print("\r" + Fore.RED + f"Unexpected error: {e}")
            return ""


# Example usage
if __name__ == "__main__":
    result = listen()
    if result:
        print(Fore.GREEN + f"User: {result}")
    else:
        print(Fore.YELLOW + "No speech detected or couldn't recognize.")