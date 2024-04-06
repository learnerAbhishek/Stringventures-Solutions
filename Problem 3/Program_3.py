from transformers import pipeline

def generate_text(input_prompt):
    """
    Generate a short paragraph of text based on the input prompt using Hugging Face transformers.

    Args:
        input_prompt (str): The input prompt to generate text.

    Returns:
        str: The generated text.
    """
    # Load the text generation pipeline
    text_generator = pipeline("text-generation", model="distilgpt2")

    # Generate text based on the input prompt
    generated_text = text_generator(input_prompt, max_length=200, num_return_sequences=1, temperature=0.9)

    # Extract and return the generated text
    return generated_text[0]['generated_text']

# Example usage:
input_prompt = "Assume you a maths teacher and you have to motivate the students to learn math"
generated_text = generate_text(input_prompt)
with open("output.txt", "w") as f:
    f.write(f"Input prompt: {input_prompt}\n")
    f.write(f"Generated text: {generated_text}\n")
print(generated_text)

