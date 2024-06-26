import openai
import CallACPFunctions
import re
# Task publishers find suitable workers based on their own task requirements

openai.api_key = "OPENAI_KEY"

def gpt_call(messages=""):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=256,
        temperature=0

    )
    print(response)

    message_content = response['choices'][0]['message']['content']
    # Extract content within quotes using regular expressions

    # If there are multiple matches, they are stored in a list
    matches = re.findall(r'"([^"]*)"|\'([^\']*)\'', message_content)
    # Get matching content
    results = [match[0] or match[1] for match in matches]
    print("The choose results are:", results)
    return results



def extract_ability(user_info_array):
    user_abilities = [user_info[2] for user_info in user_info_array]

    return user_abilities

def extract_requirement(task_info_array):

    task_requirements = [task_info[1] for task_info in task_info_array]

    return task_requirements



def owner_search_worker(requirement,workers):
    print("This is test ")
    messages = [
        {'role': 'system',
         'content': 'I am a smart agent and I can connect the smart contract with an agent and do some interaction to help you do some tasks on Ethereum.'},
        {'role': 'system',
         'content': 'When I give you a task, you should help me find a suitable worker to do my task.'},
    ]
    messages.append({'role': 'user',
                     'content': f"This is my task requirement {requirement}. This is worker list {workers}. Which worker is suitable to do my task? And tell me who theworker is"})
    return (gpt_call(messages))


def find_worker_from_requirement(task_requirement):
    task_requirement = task_requirement
    user_info_array = CallACPFunctions.getUsers()
    user_abilities= extract_ability(user_info_array)
    # Print the extracted user_abilities value
    # print("User Abilities:", user_abilities)
    users=user_abilities
    generated_texts = owner_search_worker(task_requirement, users)
    # print(generated_texts)

    for generated_text in generated_texts:
        user_index = user_abilities.index(generated_text) if generated_text in user_abilities else -1
        if user_index != -1:
            print(f"The generated text is found in user_abilities at index {user_index}.")

            user_info_indices = [idx for idx, user_info in enumerate(user_info_array) if user_abilities[user_index] in user_info]

            if user_info_indices:
                print(f"The user_abilities{user_index} is found in user_info_array at index {user_info_indices}.")
                for user_info_index in user_info_indices:
                    # Print all the information in user_info_array for the task corresponding to user_abilities[user_index]
                    user_info = user_info_array[user_info_index]
                    print(f"The user_info corresponding to user_abilities{user_index} is:")
                    print(user_info)
                    print(user_info[4])  # print worker address
                    return user_info[4]
            else:
                print("No matching user found in user_info_array.")
                return None
        else:
            print("The generated text is not found in user_abilities.")
            return None



# if __name__ == '__main__':
#     # Test
#     user_info_array = CallACPFunctions.get_users()
#     task_info_array = CallACPFunctions.get_tasks()
#
#     user_abilities= extract_ability(user_info_array)
#     task_requirements=extract_requirement(task_info_array)
#
#     # Print the extracted ability and requirement values
#     print("User Abilities:", user_abilities)
#     print("Task Requirements:", task_requirements)
#
#     workers=user_abilities
#     # print(owner_search_worker(task_requirement,workers))
#     params = {"id": 6}# Test case,
#     task_requirement=CallACPFunctions.getTaskWithID(params)
#     worker=find_worker_from_requirement(task_requirement) # Obtain the worker address, which can use to deploy for owner task.
#     print("print worker.........")
#     print(worker)

