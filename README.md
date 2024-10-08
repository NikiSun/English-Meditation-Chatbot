# English-Meditation-Chatbot
This is a chatbot in English for Meditation. Two fine-tuned LLMs has been used here.

# Dataset
76 English meditation scripts were collected from publicy avaiable website.
Four classifications: Positive Thinking, Relieving Loneliness, Relieving Stress and Anxiety, Helping Sleep.
Each classifications have 14 short scripts (200-1000 words) and 5 long scripts.(1000-2000 words)
All 76 scripts were carefully checked and added SSML tags for later fine-tuning.

# ChatGPT
Google Text-to-Speech and Speech-to-Text API were applied.
Scripts generator Model was fine-tuned in OpenAI API by Dataset. (As a reference, training loss of mine is 0.2159.)
MeditAI.py is the meditation chatbot.
Updated_Model.py is the improved chatbot by adding pause to each punctuation and check the number of words. 

# LLaMA
FineTune.py was used to fine tune LLaMA-2 model from HuggingFace. (my fine-tuned open-source model: Niki-1115/llama-meditation-optimized-4. Training Loss:2.69)
MeditAI.py is the same but based on fined-tuned LLaMA model.

# Evaluation
After processing by CleanScripts.py,
the generted scripts were evaluated by BLEU, ROUGE, METEOR, and BERTScore.
