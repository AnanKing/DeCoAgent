import openai
import re
import TestACPwithGPT
import Verificator
import numpy as np
openai.api_key = "OPENAI_KEY"


def process_input_string_to_actions_sequence(input_string):
    # Use regular expressions to match dots after numbers and dots after done and replace them with empty strings
    processed_string = re.sub(r'\d+\.', '', input_string)
    processed_string = re.sub(r'done\.', 'done', processed_string)

    # Split string into array using regular expression
    result_array = re.findall(r'\S.*?(?=\d+\.)|\S.*?(?=done)|\S.*?(?=$)', processed_string)

    # Remove commas and spaces that may be present at the end of each element
    result_array = [element.rstrip(', ') for element in result_array]

    return result_array



def gpt_call(messages=""):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=256,
        temperature=0,
        logprobs=True, # Get log probabilities for the response
        top_logprobs=1
    )
    print(response)

    logprobs = response['choices'][0].get('logprobs', {})
    # print("Log probabilities:", logprobs)
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

    print(f"The steps are : {response['choices'][0]['message']['content']}")
    return response['choices'][0]['message']['content']


if __name__ == '__main__':
    messages = [
        {'role': 'system', 'content': 'I am an AI agent operating in the collaboration platform which runs a smart contract on Ethereum. This smart contract offers a wide array of functions, each serving specific purpose: Agent Registered Function allows agents to become a user in smart contracts. Query Ability Function allows the user to query details information of user by input ability. The Get Task With ID Function allows the user to obtain details information about the task by inputting the task ID. Post Task Function allows the user to post their task to smart contract. Deploy Task Function allows the user to deploy a task to another user in the smart contract. Request Task Function allows the user to request a task from the task owner to become a worker in the smart contract. Rating Worker Function allows the user to rate a worker of the task. The Get Users Function allows the user to obtain information about all users. The Get Tasks Function allows the user to obtain information about all tasks.'},
        {'role': 'system', 'content': 'You can ask me to do various tasks and I will tell you the sequence of actions I would do to accomplish your task.'},
        {'role': 'user', 'content': 'How would you join the smart contract by input ability is {I am a teacher} and contact is {test001@gmail.com} ?'},
        {'role': 'assistant','content': '1. call Agent Registered Function with ability is {I am a teacher} and contact is {test001@gmail.com}, 2. done.'},
        {'role': 'user', 'content': 'How would you post a task in smart contract by input requirement is {I have a question} and price is {10}?'},
        {'role': 'assistant', 'content': '1. call Post Task Function with requirement is {I have a question} and price is {10}, 2. done.'},
        {'role': 'user', 'content': 'How would you query details information of user by input ability is {math}?'},
        # {'role': 'assistant', 'content': '1. call  Query Ability Function with ability is {math}, 2. done.'},
        # {'role': 'user', 'content': 'How would you deploy a task to other user in smart contract by input ID is {1}?'},
        # {'role': 'assistant', 'content': '1. call  Deploy Task Function with ID is {1}, 2. done.'},
        # {'role': 'user','content': 'How would you request a task to from task owner become a worker in smart contract by input ID is {1}?'},
        # {'role': 'assistant', 'content': '1. call Request Task Function with ID is {1}, 2. done.'},
        # {'role': 'user', 'content': 'How would you get the information of task by input task ID is {8}? '},
        # {'role': 'assistant', 'content': '1. call Get Task With ID Function with task ID is {9}, 2. done.'}, #只给一个输入的例子就能生成一样的
        # {'role': 'user', 'content': 'How would you get the information of all tasks?'}
        # {'role': 'user', 'content': 'How would you get the information of all users?'}
        # {'role': 'user', 'content': 'How would you rate a worker by input task ID is {1} and score is {5}?'}
    ]

    # generate the workflow
    response_LLMs=gpt_call(messages=messages)
    print(response_LLMs)

    # verfication the workflow
    verfication_result = Verificator.verificator(messages, response_LLMs)
    if verfication_result == 'True':
        # parase the wotrkflow to each subtask
        call_steps=process_input_string_to_actions_sequence(response_LLMs)
        # print(call_steps)

        for call in call_steps:
            if call == 'done':
                print("The program has been completed......")
            else:
                TestACPwithGPT.run_conversation_call_functions(call)





