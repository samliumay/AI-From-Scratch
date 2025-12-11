"""
Interactive UstaModel Text Generator
Run this script to chat with the trained UstaModel
"""

from basicLLM import load_model, generate_text, MODEL_CONFIG

def main():
    # Load the model
    print("Loading UstaModel...")
    try:
        model = load_model('u_model.pth')
        print("Model loaded successfully!")
        print(f"- Vocab size: {MODEL_CONFIG['vocab_size']}")
        print(f"- Context length: {MODEL_CONFIG['context_length']}")
        print(f"- Layers: {MODEL_CONFIG['num_layers']}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Interactive loop
    print("\n" + "="*50)
    print("UstaModel Interactive Chat")
    print("="*50)
    print("Tips:")
    print("- Type 'exit' to quit")
    print("- Keep inputs short (max ~15 words)")
    print("- Use words from the vocabulary for best results")
    print("- Do not expext much. Mainly for fun!")
    print("- Training dataset generally included capitals, Eurpian Capitals, general english and small amount of math. So basic prompts like: 'Capital, Paris, London, Paris is, Lisbon' will give at least an answer." )
    print("="*50 + "\n")
    
    max_input_words = 15
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for exit
            if user_input.lower() == 'exit':
                print("\nGoodbye!")
                break
            
            # Skip empty inputs
            if not user_input:
                continue
            
            # Check input length
            word_count = len(user_input.split())
            if word_count > max_input_words:
                print(f"Input too long! Please use {max_input_words} words or less.")
                print(f"(You used {word_count} words)")
                continue
            
            # Generate response
            print("AI: ", end="", flush=True)
            response = generate_text(
                model, 
                user_input, 
                max_tokens=30, 
                temperature=0.8
            )
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! ")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again with a different input.\n")

if __name__ == "__main__":
    main()
