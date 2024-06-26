import openai
import numpy as np

# Set your OpenAI API key
openai.api_key = "OPENAI_KEY"

def verificator(prompt1,result1):
    prompt2 = (
    f"Here is a prompt and its generated response:\n\n"
    f"Prompt: {prompt1}\n\n"
    f"Generated Response: {result1}\n\n"
    f"Is the generated response correct? Answer 'True' or 'False'."
)
    try:
        response2 = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt2}
        ],
        max_tokens=5,
        temperature=0,
        logprobs=True,
        top_logprobs=1
    )

        validation_result = response2['choices'][0]['message']['content'].strip()
        logprobs = response2['choices'][0].get('logprobs', {})
        if logprobs and 'content' in logprobs:
            token_logprobs = [entry['logprob'] for entry in logprobs['content']]
            if token_logprobs:  # Ensure it's not empty
                sum_logprobs = sum(token_logprobs)
                print(sum_logprobs)
                linear_prob = np.round(np.exp(sum_logprobs) * 100, 2)
                print("Linear probability:", linear_prob)
                print("Token log probabilities:", token_logprobs, "Sum log probabilities:", sum_logprobs)
            else:
                print("Token log probabilities are empty.")
        else:
            print("Log probabilities do not contain 'content'.")

        # Print validation result
        print("Validation Result:", validation_result)
        # print(response2['choices'])


    except openai.error.OpenAIError as e:
        print("An error occurred:", e)

    return validation_result
