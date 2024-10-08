import re

# Define the file paths
file_path = r'' # Path to the generated SSML file
cleaned_output_file_path = r'' # Path to save the cleaned content

# Read the content from the generated SSML file
with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# Use a regular expression to remove all <> and the content within them
cleaned_content = re.sub(r'<.*?>', '', content)

# Save the cleaned content to a new file
with open(cleaned_output_file_path, 'w', encoding='utf-8') as cleaned_file:
    cleaned_file.write(cleaned_content)

print(f"Cleaned content saved to: {cleaned_output_file_path}")
