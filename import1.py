from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


def chat():
    print("Simple Chat System (type 'quit' to exit)")
    while True:
        # Get user input
        user_input = input("You: ")

        if user_input.lower() == 'quit':
            break

        # Generate response
        input_ids = tokenizer.encode(user_input, return_tensors="pt")
        output = model.generate(
            input_ids,
            max_length=100,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode and print response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"AI: {response[len(user_input):]}")


if __name__ == "__main__":
    chat()
