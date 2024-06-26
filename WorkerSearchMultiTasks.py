import openai
import CallACPFunctions
import re
# Workers find suitable tasks based on their abilities

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

    results = [match[0] or match[1] for match in matches]
    print("The choose results are:", results)
    return results



def extract_requirement(task_info_array):

    task_requirements = [task_info[1] for task_info in task_info_array]

    return task_requirements


def work_search_task(ability,tasks):
    messages = [
        {'role': 'system',
         'content': 'I am a smart agent and I can connect the smart contract with an agent and do some interaction to help you do some tasks on Ethereum.'},
        {'role': 'system',
         'content': 'When I give you a user ability, you should help me find a suitable task for me.'},
    ]
    messages.append({'role': 'user', 'content': f"This is my ability {ability}. This is task list {tasks}. Which task do you think I'm suitable for? And tell me which task is"})
    return(gpt_call(messages))



def find_owner_from_task(ability):
    ability = ability
    task_info_array = CallACPFunctions.getTasks()
    task_requirements = extract_requirement(task_info_array)

    # Print the extracted requirement value
    print("Task Requirements:", task_requirements)
    tasks = task_requirements
    generated_texts = work_search_task(ability, tasks)
    print(generated_texts)
    for generated_text in generated_texts:
        task_index = task_requirements.index(generated_text) if generated_text in task_requirements else -1
        if task_index != -1:
            print(f"The generated text is found in task_requirements at index {task_index}.")

            task_info_indices = [idx for idx, task_info in enumerate(task_info_array) if
                                 task_requirements[task_index] in task_info]

            if task_info_indices:
                print(f"The task_requirements{task_index} is found in task_info_array at index {task_info_indices}.")

                for task_info_index in task_info_indices:

                    task_info = task_info_array[task_info_index]
                    print(f"The task_info corresponding to task_requirements{task_index} is:")
                    print(task_info)
                    print(task_info[0])  # print the task ID
                    return task_info[0]
            else:
                print("No matching task found in task_info_array.")
                return None

        else:
            print("The generated text is not found in task_requirements.")
            return None




# if __name__ == '__main__':
#     # Test
#     account_address = "YOUR_ACCOUNT_ADDRESS"
#     ability=CallACPFunctions.get_ability(account_address)
#     print("The ability is:")
#     print(ability)
#     owner=find_owner_from_task(ability)
#     print("The owner address is:")
#     print(owner)
#
#     owner_contact=CallACPFunctions.get_contact(owner)
#     print(owner_contact)








