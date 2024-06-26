import openai
import json
import MappingFunctions

openai.api_key = "OPENAI_KEY"

tools = [
    {
        "type": "function",
        "function": {
            "name": "AgentRegisteredFunction",
            "description": "Agent registered in System as a participant",
            "parameters": {
                "type": "object",
                "properties": {
                    "ability": {
                        "type": "string",
                        "description": "The ability of user"
                    },
                    "contact": {
                        "type": "string",
                        "description": "The contact of agent, such as email"
                    }
                },
                "required": ["address","ability","contact"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "QueryAbilityFunction",
            "description": "Query the ability of agent in System",
            "parameters": {
                "type": "object",
                "properties": {
                    "ability": {
                        "type": "string",
                        "description": "The ability of agent"
                    }
                },
                "required": ["ability"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "getTaskWithIDFunction",
            "description": "Get the task information with task id in System",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "The id of task"
                    }
                },
                "required": ["id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "getTasksFunction",
            "description": "Get all the task information in System",
            "parameters": {
               "type": "object",
                 "properties": {
                 },
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "getUsersFunction",
            "description": "Get all the user information in System",
            "parameters": {
                "type": "object",
                "properties": {
                },
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "getUserByAddressFunction",
            "description": "Get all the user information in System",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {
                        "type": "string",
                        "description": "The address of user"
                    }
                },
                "required": ["address"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "PostTaskFunction",
            "description": "The owner of task is post the task into system",
            "parameters": {
                "type": "object",
                "properties": {
                    "requirement": {
                        "type": "string",
                        "description": "The requirement of task"
                    },
                    "price": {
                        "type": "integer",
                        "description": "The price of task"
                    }
                },
                "required": ["requirement","price"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "DeployTaskFunction",
            "description": "The owner of task is deployed task in the system",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "The id of the task"
                    },
                },
                "required": ["id"]
            }
        }
    },
{
        "type": "function",
        "function": {
            "name": "RequestTaskFunction",
            "description": "The worker to request a task from task owner become to worker in the system",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "The id of user"
                    },
                },
                "required": ["id"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "RatingWorkerFunction",
            "description": "The owner of task is rating the worker",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "The ID of the task"
                    },
                    "rating": {
                        "type": "integer",
                        "description": "The rating value of the worker"
                    }
                },
                "required": ["id", "rating"]
            }
        }
    }
]



def run_conversation_call_functions(contents):
    messages = [{"role": "user", "content": f"How to {contents} in system"}]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    # response_message = response['choices'][0]['message']
    # print(response_message)

    response_message = response.choices[0].message
    # print(response_message)
    # tool_calls = response_message['tool_calls']
    tool_calls = response_message.get('tool_calls',[])
    while len(response_message)>0:
        if tool_calls:
            availabe_functions = {
                'AgentRegisteredFunction': MappingFunctions.AgentRegisteredFunction,
                'QueryAbilityFunction': MappingFunctions.QueryAbilityFunction,
                'PostTaskFunction': MappingFunctions.PostTaskFunction,
                'DeployTaskFunction': MappingFunctions.DeployTaskFunction,
                'RequestTaskFunction': MappingFunctions.RequestTaskFunction,
                'RatingWorkerFunction': MappingFunctions.RatingWorkerFunction,
                'getTaskWithIDFunction': MappingFunctions.getTaskWithIDFunction,
                'getTasksFunction': MappingFunctions.getTasksFunction,
                'getUsersFunction': MappingFunctions.getUsersFunction,
                'getUserByAddressFunction': MappingFunctions.getUserByAddressFunction,
            }
            messages.append(response_message)
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = availabe_functions[function_name]
                if function_to_call is None:
                    print(f"Function {function_name} not found.")
                    continue
                function_args = json.loads(tool_call.function.arguments)
                print(function_args)
                function_response = function_to_call(**function_args)
                print(function_response)
                if function_response is None:
                    function_response = "No response"  # Set a default value, make sure it is a string
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                })
                # Check if 'content' is null and set it to an empty string
            # if response_message['content'] is None:
            #     response_message['content'] = ''
            second_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-1106",
                messages=messages,
            )
            return second_response



'''
# Test Case
# tx = "call User Registered Function"
# print(run_conversation_call_functions(tx))

# tx = "call Query Ability Function"
# print(run_conversation_call_functions(tx))

# tx = "call Post Task Function"
# print(run_conversation_call_functions(tx))

# tx = "Call the Post Task Function with requirement is {I have a question} and price is {10}"
# print(run_conversation_call_functions(tx))

# tx = "call Request Task Function with ID is {1}"
# print(run_conversation_call_functions(tx))

tx = "call  Deploy Task Function with ID is {1}"
print(run_conversation_call_functions(tx))

# tx = "call Rating Worker Function"
# print(run_conversation_call_functions(tx))

# tx = "call get Task With ID Function with ID is {8}"
# print(run_conversation_call_functions(tx))

# tx = "call get Task Function without ID"
# print(run_conversation_call_functions(tx))

# tx = "call get User Function"
# print(run_conversation_call_functions(tx))
# call Get Task With ID Function with 9

# tx = "call Rating Worker Function with task ID is {6} and score is {5}"
# print(run_conversation_call_functions(tx))

# tx = "call get User By Address Function with address is {0x3447....................}"
# print(run_conversation_call_functions(tx))

'''
