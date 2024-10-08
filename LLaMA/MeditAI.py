import os
import pyaudio
import io
from pydub import AudioSegment
import speech_recognition as speechtotext
from google.cloud import texttospeech
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Set Google Cloud credentials for Speech-to-Text (STT)
os.environ["GOOGLE_APPLICATION_CREDENTIALS_STT"] = r"" # Add your Google Cloud STT credentials here

# Set Google Cloud credentials for Text-to-Speech (TTS)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"" # Add your Google Cloud TTS credentials here

# Load your fine-tuned LLaMA model with PEFT configuration and tokenizer from Hugging Face
config = PeftConfig.from_pretrained("") # Add the path to your PEFT configuration file

# Load base LLaMA model
base_model = AutoModelForCausalLM.from_pretrained("") # Add the path to your base LLaMA model
 
# Load tokenizer from the fine-tuned model to ensure compatibility
tokenizer = AutoTokenizer.from_pretrained("") # Add the path to your tokenizer

# Resize base model's token embeddings to match the tokenizer's vocabulary size
base_model.resize_token_embeddings(len(tokenizer))

# Load the PEFT model on top of the resized base model
model = PeftModel.from_pretrained(base_model, "") # Add the path to your fine-tuned PEFT model




# Speech-to-text processing
def recognize_speech():
    r = speechtotext.Recognizer()
    with speechtotext.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)

    try:
        recognized_text = r.recognize_google_cloud(audio, credentials_json=os.getenv("GOOGLE_APPLICATION_CREDENTIALS_STT"))
        print("You said:", recognized_text)
        return recognized_text
    except speechtotext.UnknownValueError:
        audio_understand_instruction = ("I could not understand you, would you like to repeat that?")
        print(audio_understand_instruction)
        audio_data = text_to_speech_ssml(audio_understand_instruction)
        play_audio(audio_data)
        return None
    except speechtotext.RequestError as e:
        print(f"Could not request results from Google Cloud Speech service: {e}")
        return None

# Function to determine the classification based on user input
def determine_classification(user_input):
    classification_keywords = {
        "Positive Thinking": ["motivation", "inspiration", "encouragement", "positive", "happy"],
        "Reliefing Loneliness": ["lonely", "isolation", "alone", "empty", "isolated"],
        "Reliefing Stress and Anxiety": ["stress", "anxiety", "tension", "nervous", "worry"],
        "Helping Sleep": ["sleep", "awake", "rest", "stay up late", "sleeplessness"]
    }

    for classification, keywords in classification_keywords.items():
        for keyword in keywords:
            if keyword in user_input.lower():
                return classification
    return None

# Function to determine short or long practice based on user input
def determine_length(user_input):
    if "short" in user_input.lower():
        return "short"
    elif "long" in user_input.lower():
        return "long"
    else:
        return None

# Function to generate response using the fine-tuned PEFT model
def generate_response(user_input, classification, practice_type):
    prompt = (
        f"User input: {user_input}\n"
        f"Assistant: I understand. You are seeking a {practice_type} meditation practice for {classification}.\n"
        "Here is a guided meditation practice for you:"
    )

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate a response using the PEFT model
    outputs = model.generate(inputs['input_ids'], max_length=150)
    
    # Decode the response to text
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Append the response to the SSML file
    with open(r"", "a", encoding="utf-8") as file: # Add the path to your SSML file
        file.write(response_text + "\n\n\n\n")
    
    return response_text

# Text-to-speech conversion using Google Cloud TTS
def text_to_speech_ssml(ssml_text):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    return response.audio_content

# Function to play audio using pyaudio and pydub
def play_audio(audio_data):
    audio_io = io.BytesIO(audio_data)
    audio_segment = AudioSegment.from_file(audio_io, format="wav")

    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(audio_segment.sample_width),
                    channels=audio_segment.channels,
                    rate=audio_segment.frame_rate,
                    output=True)
    stream.write(audio_segment.raw_data)

    stream.stop_stream()
    stream.close()
    p.terminate()

# Example usage
opening_message = "Hey there! Welcome to Medit AI, your personal mindfulness companion. I'm here to support you through four different types of meditation: Positive Thinking, Reliefing Loneliness, Reliefing Stress and Anxiety, and Helping Sleep. How are you feeling today? Let me know which one you need right now, and I'll guide you through it!"
print(opening_message)

audio_data = text_to_speech_ssml(opening_message)
play_audio(audio_data)

while True:
    classification = None
    length = None

    while True:
        user_input = recognize_speech()
        if user_input:
            user_input = user_input.strip().lower()
            if user_input == "no":
                exit_message = "Well, my work here is done. Have a great day!"
                audio_data = text_to_speech_ssml(exit_message)
                play_audio(audio_data)
                print(f"{exit_message}\n Conversation ended.")
                exit(0)

            if not classification:
                classification = determine_classification(user_input)
                if classification:
                    print(f"Detected classification: {classification}")
                    length_instruction = f"I understand. Would you like a short or long meditation practice for {classification}?"
                    print(length_instruction)
                    audio_data = text_to_speech_ssml(length_instruction)
                    play_audio(audio_data)
                    continue
                else:
                    classification_instruction = "Please describe what you're looking for, such as 'I need help with stress.'"
                    print(classification_instruction)
                    audio_data = text_to_speech_ssml(classification_instruction)
                    play_audio(audio_data)
                    continue

            if classification and not length:
                length = determine_length(user_input)
                if length:
                    print(f"Detected practice length: {length}")
                else:
                    length_instruction_repetition = "Couldn't detect meditation length. Please try again."
                    print(length_instruction_repetition)
                    audio_data = text_to_speech_ssml(length_instruction_repetition)
                    play_audio(audio_data)
                    continue

            if classification and length:
                confirmation_message = f"Got it! You want a {length} meditation practice for {classification}. Let me prepare the right practice for you."
                print(confirmation_message)
                audio_data = text_to_speech_ssml(confirmation_message)
                play_audio(audio_data)

                # Generate suitable scripts and convert them to speech
                response_text = generate_response(user_input, classification, length)
                print("AI Response:", response_text)
                audio_data = text_to_speech_ssml(response_text)
                play_audio(audio_data)
                break  # Break out of the inner loop

    other_problems_instruction = "Do you have any other needs?"
    print(other_problems_instruction)
    audio_data = text_to_speech_ssml(other_problems_instruction)
    play_audio(audio_data)
