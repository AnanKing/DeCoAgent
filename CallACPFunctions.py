import jsonschema
from web3 import Web3
import json
import OwnerSearchMultiWorkers
import WorkerSearchMultiTasks

# web3 instance, Needs to be replaced with the URL of GANACHE
w3 = Web3(Web3.HTTPProvider("YOUR_RPC_SERVER_IN_GANACHE"))
# Test is connected Ganache
print(f'Is connected Ganache : {w3.is_connected()}')

# Needs to be replaced with your smart contract ABI
with open('YOUR_CONTRACT_ABI_FILE_JSON', 'r') as f:
    contract_abi = json.load(f)

# Need to be replaced with the address of the smart contract
contract_address = "YOUR_SMART_CONTRACT_ADDRESS"
check_address = Web3.to_checksum_address(contract_address)

contract = w3.eth.contract(address=check_address, abi=contract_abi)

# Need to be replaced with the your address
account = "YOUR_ADDRESS_IN_GANACHE"
# Need to be replaced with the your private key
private_key = "YOUR_ACCOUNT_PRIVATE_KEY"

def get_task_arguments():
    with open("TaskPost.json", 'r') as file:
        data = json.load(file)

    if isinstance(data, list) and len(data) > 0:
        task = data[0]
        if 'function' in task and 'parameters' in task['function']:
            parameters = task['function']['parameters']
            if 'properties' in parameters:
                properties = parameters['properties']

                # Get the value of "requirement" and "price" fields
                requirement_value = properties.get('requirement', {}).get('value')
                price_value = properties.get('price', {}).get('value')

                return requirement_value, price_value
    return None, None

def get_agent_arguments():
    with open("UserRegister.json", 'r') as file:
        data = json.load(file)

    if isinstance(data, list) and len(data) > 0:
        agent = data[0]
        if 'function' in agent and 'parameters' in agent['function']:
            parameters = agent['function']['parameters']
            if 'properties' in parameters:
                properties = parameters['properties']

                # Get the value of "ability" and "contact" fields
                ability_value = properties.get('ability', {}).get('value')
                contact_value = properties.get('contact', {}).get('value')

                return ability_value, contact_value
    return None, None

register_schema = {
    "type": "object",
    "properties": {
        "ability": {"type": "string"},
        "contact": {"type": "string"},
    },
    "required": ["ability", "contact"]
}

def register(params):
    jsonschema.validate(params, register_schema)

    # Destructuring parameters
    # Get agent information from local user files
    ability,contact=get_agent_arguments()
    transaction_data = contract.functions.register(ability, contact).build_transaction({
        'from': account,
        'gas': 1000000,
        'gasPrice': w3.to_wei('10', 'gwei'),
        'nonce': w3.eth.get_transaction_count(account)
    })
    # Sign the transaction
    signed_transaction = w3.eth.account.sign_transaction(transaction_data, private_key)

    # Send the transaction
    transaction_hash = w3.eth.send_raw_transaction(signed_transaction.rawTransaction)

    print(f'Register transaction sent. Transaction Hash: {transaction_hash.hex()}')

    # Wait for the transaction to be mined
    transaction_receipt = w3.eth.wait_for_transaction_receipt(transaction_hash)

    print(transaction_receipt)
    print("AgentRegisteredFunction executed successfully")
    # Get the event logs
    logs = contract.events.UserRegistered().process_receipt(transaction_receipt)
    for log in logs:
        # print(f"User Registered: {log['args']}")
        print(log['args']['id'])
    return transaction_receipt


query_by_ability_schema = {
    "type": "object",
    "properties": {
        "ability": {"type": "string"},
    },
    "required": ["ability"]
}

def query_by_ability(params):
    jsonschema.validate(params, query_by_ability_schema)

    # Destructuring parameters
    ability = params["ability"]

    # Call the queryByAbility function
    returned_users = contract.functions.queryByAbility(ability).call()

    # Print the returned users
    for user in returned_users:
        user_id = user[0]
        is_free = user[1]
        user_ability = user[2]
        user_contact = user[3]
        user_address = user[4]
        user_rating = user[5]
        print(f"User ID: {user_id}, Is Free: {is_free},Ability: {user_ability}, Contact: {user_contact},Address: {user_address}, Rating: {user_rating}")

    return returned_users


post_task_schema = {
    "type": "object",
    "properties": {
        "requirement": {"type": "string"},
        "price": {"type": "integer"},
    },
    "required": ["requirement","price"]
}

def post_task(params):
    jsonschema.validate(params, post_task_schema)

    # Destructuring parameters
    # Get parameters from Function Calling
    requirement = params["requirement"]
    price = params["price"]
    # Get parameters from local JSON file
    # requirement,  price = get_task_arguments()

    transaction_data = contract.functions.postTask(requirement, price).build_transaction({
        'from': account,
        'gas': 1000000,
        'gasPrice': w3.to_wei('10', 'gwei'),
        'nonce': w3.eth.get_transaction_count(account)
    })
    # Sign the transaction
    signed_transaction = w3.eth.account.sign_transaction(transaction_data, private_key)

    # Send the transaction
    transaction_hash = w3.eth.send_raw_transaction(signed_transaction.rawTransaction)

    print(f'post task transaction sent. Transaction Hash: {transaction_hash.hex()}')

    # Wait for the transaction to be mined
    transaction_receipt = w3.eth.wait_for_transaction_receipt(transaction_hash)

    print(transaction_receipt)
    print("PostTaskFunction executed successfully")

    logs = contract.events.TaskPosted().process_receipt(transaction_receipt)
    for log in logs:
        print(f"post Task: {log['args']}")
        print(log['args']['id'])

    return transaction_receipt

deploy_task_schema = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
    },
    "required": ["id"]
}

# This function is used to 'task owner driven' to deploy task
def deploy_task(params):
    jsonschema.validate(params, deploy_task_schema)

    # Destructuring parameters
    id = params["id"]
    # Get parameters from Function Calling
    params = {"id": id}
    task_requirement = get_task_with_ID(params)  # Obtain the task by call get_task_with_ID() function
    address = OwnerSearchMultiWorkers.find_worker_from_requirement(task_requirement)  # Obtain the worker address that can be used to deploy tasks to the worker

    print(address)

    transaction_data = contract.functions.deployTask(id,address).build_transaction({
        'from': account,
        'gas': 1000000,
        'gasPrice': w3.to_wei('10', 'gwei'),
        'nonce': w3.eth.get_transaction_count(account)
    })
    # Sign the transaction
    signed_transaction = w3.eth.account.sign_transaction(transaction_data, private_key)

    # Send the transaction
    transaction_hash = w3.eth.send_raw_transaction(signed_transaction.rawTransaction)

    print(f'Deploy task transaction sent. Transaction Hash: {transaction_hash.hex()}')

    # Wait for the transaction to be mined
    transaction_receipt = w3.eth.wait_for_transaction_receipt(transaction_hash)

    print(transaction_receipt)
    print("DeployTaskFunction executed successfully")

    logs = contract.events.TaskDeployed().process_receipt(transaction_receipt)
    for log in logs:
        print(f"Task Deployed: {log['args']}")
        print(log['args']['id'])

    return transaction_receipt

request_task_schema = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
    },
    "required": ["id"]
}

# This function is used to 'worker driven' to request task to deploy
def request_task(params):
    jsonschema.validate(params, request_task_schema)

    # Destructuring parameters
    id = params["id"]
    # Get parameters from Function Calling
    params = {"id":id}
    user_ability=get_ability_with_ID(params)# Obtain the task by call get_ability_with_ID() function
    task_ID=WorkerSearchMultiTasks.find_owner_from_task(user_ability)  # Obtain the task ID that worker can become a worker

    print(task_ID)

    transaction_data = contract.functions.requestTask(id,task_ID).build_transaction({
        'from': account,
        'gas': 1000000,
        'gasPrice': w3.to_wei('10', 'gwei'),
        'nonce': w3.eth.get_transaction_count(account)
    })
    # Sign the transaction
    signed_transaction = w3.eth.account.sign_transaction(transaction_data, private_key)

    # Send the transaction
    transaction_hash = w3.eth.send_raw_transaction(signed_transaction.rawTransaction)

    print(f'Request a task transaction sent. Transaction Hash: {transaction_hash.hex()}')

    # Wait for the transaction to be mined
    transaction_receipt = w3.eth.wait_for_transaction_receipt(transaction_hash)

    print(transaction_receipt)
    print("RequestTaskFunction executed successfully with worker")

    logs = contract.events.TaskDeployed().process_receipt(transaction_receipt)
    for log in logs:
        print(f"Task Deployed: {log['args']}")
        print(log['args']['id'])

    return transaction_receipt


deploy_task_with_worker_schema = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "address": {"type": "string"},
    },
    "required": ["id"]
}

# This function is used to 'worker driven' to send their address to the task owner to deploy the task
def deploy_task_with_worker(params):
    jsonschema.validate(params, deploy_task_with_worker_schema)

    # Destructuring parameters
    id = params["id"]
    address = params["address"]

    transaction_data = contract.functions.deployTask(id,address).build_transaction({
        'from': account,
        'gas': 1000000,
        'gasPrice': w3.to_wei('10', 'gwei'),
        'nonce': w3.eth.get_transaction_count(account)
    })
    # Sign the transaction
    signed_transaction = w3.eth.account.sign_transaction(transaction_data, private_key)

    # Send the transaction
    transaction_hash = w3.eth.send_raw_transaction(signed_transaction.rawTransaction)

    print(f'Deploy task transaction sent. Transaction Hash: {transaction_hash.hex()}')

    # Wait for the transaction to be mined
    transaction_receipt = w3.eth.wait_for_transaction_receipt(transaction_hash)

    print(transaction_receipt)
    print("DeployTaskFunction executed successfully with worker")

    logs = contract.events.TaskDeployed().process_receipt(transaction_receipt)
    for log in logs:
        print(f"Task Deployed: {log['args']}")
        print(log['args']['id'])

    return transaction_receipt


rating_worker_schema = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "rating":{"type": "integer"},
    },
    "required": ["id","rating"]
}

def rating_worker(params):
    jsonschema.validate(params, rating_worker_schema)

    # Destructuring parameters
    id = params["id"]
    rating = params["rating"]

    transaction_data = contract.functions.ratingWorker(id, rating).build_transaction(
        {
            'from': account,
            'gas': 1000000,
            'gasPrice': w3.to_wei('10', 'gwei'),
            'nonce': w3.eth.get_transaction_count(account)
        })
    # Sign the transaction
    signed_transaction = w3.eth.account.sign_transaction(transaction_data, private_key)

    # Send the transaction
    transaction_hash = w3.eth.send_raw_transaction(signed_transaction.rawTransaction)

    print(f'Rating worker transaction sent. Transaction Hash: {transaction_hash.hex()}')

    # Wait for the transaction to be mined
    transaction_receipt = w3.eth.wait_for_transaction_receipt(transaction_hash)

    print(transaction_receipt)
    print("RatingWorkerFunction executed successfully")

    logs = contract.events.WorkerRated().process_receipt(transaction_receipt)
    for log in logs:
        print(f"Worker Rated: {log['args']}")
        print(log['args']['id'])

    return transaction_receipt


get_task_with_ID_schema = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
    },
    "required": ["id"]
}

def get_task_with_ID(params):
    jsonschema.validate(params, get_task_with_ID_schema)

    # Destructuring parameters
    id = params["id"]

    returned_tasks = contract.functions.getTaskWithID(id).call()
    print(returned_tasks)
    task_id = returned_tasks[0]
    task_requirement = returned_tasks[1]
    task_owner = returned_tasks[2]
    task_worker = returned_tasks[3]
    task_isDeployed = returned_tasks[4]
    task_price = returned_tasks[5]
    task_isPay = returned_tasks[6]
    print(
        f"Task ID: {task_id}, task requirement: {task_requirement},task owner: {task_owner}, task worker: {task_worker},task is Deployed: {task_isDeployed}, task price: {task_price},task is Pay: {task_isPay}")
    return returned_tasks[1]

get_user_with_ID_schema = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
    },
    "required": ["id"]
}
def get_ability_with_ID(params):
    jsonschema.validate(params, get_user_with_ID_schema)

    # Destructuring parameters
    id = params["id"]

    returned_users = contract.functions.getUserById(id).call()
    print(returned_users)
    user_id = returned_users[0]
    user_isFree = returned_users[1]
    user_ability = returned_users[2]
    user_contract = returned_users[3]
    user_address = returned_users[4]
    user_rating = returned_users[5]
    print(
        f"User ID: {user_id}, user isFree: {user_isFree},user ability: {user_ability}, user contract: {user_contract},user address: {user_address}, user rating: {user_rating}")
    return returned_users[2]


get_tasks_schema = {
    "type": "object",
    # "properties": {
    #     "id": {"type": "integer"},
    # },
    # "required": ["id"]
}

type_mapping = {
    'integer': int,
    'string': str,
    'bool': bool
}
def validate_task(task, parameters):
    properties = parameters.get('properties', {})
    required_fields = parameters.get('required', [])

    # Check if all required fields are present
    if not all(field in task for field in required_fields):
        return False

    for key, value in task.items():
        if key in properties:
            prop_info = properties[key]

            if isinstance(prop_info, dict):
                expected_type = prop_info.get('type')

                if 'value' not in value or not isinstance(value['value'], type_mapping.get(expected_type, str)):
                    return False
            else:
                return False

    return True


def validate_user(user, parameters):
    properties = parameters.get('properties', {})
    required_fields = parameters.get('required', [])

    # Check if all required fields are present
    if not all(field in user for field in required_fields):
        return False

    for key, value in user.items():
        if key in properties:
            prop_info = properties[key]

            if isinstance(prop_info, dict):
                expected_type = prop_info.get('type')

                if 'value' not in value or not isinstance(value['value'], type_mapping.get(expected_type, str)):
                    return False
            else:
                return False

    return True

def get_tasks():
    returned_tasks = contract.functions.getTasks().call()
    # print(returned_tasks)
    # Print the returned users
    for task in returned_tasks:
        task_id = task[0]
        task_requirement = task[1]
        task_owner = task[2]
        task_worker = task[3]
        task_isDeployed = task[4]
        task_price= task[5]
        task_isPay =task[6]
        print(
            f"Task ID: {task_id}, task requirement: {task_requirement},task owner: {task_owner}, task worker: {task_worker},task is Deployed: {task_isDeployed}, task price: { task_price},task is Pay: {task_isPay}")


    #Save these tasks in local machine
    with open("DownloadTaskFromSolidity.json",'r') as file:
        data=json.load(file)
    if isinstance(data, list) and len(data) > 0:
        # Get the parameter structure of the first task
        parameters = data[0]['function']['parameters']
        if 'properties' in parameters:
            properties = parameters['properties']

            # Add each task to JSON file
            for task in returned_tasks:
                new_task = {
                    'task_id': {'value': task[0], 'type': 'integer'},
                    'requirement': {'value': task[1], 'type': 'string'},
                    'owner': {'value': task[2], 'type': 'string'},
                    'worker': {'value': task[3], 'type': 'string'},
                    'deployed': {'value': task[4], 'type': 'bool'},
                    'price': {'value': task[5], 'type': 'integer'},
                    'Pay': {'value': task[6], 'type': 'bool'}
                }

                # Verify that the task meets the requirements
                if validate_task(new_task, properties):
                    data.append({'type': 'Task', 'function': {'name': 'TaskName', 'description': 'The name of task',
                                                              'parameters': {'type': 'object', 'properties': new_task,
                                                                             'required': ['requirement', 'price']}}})

        # Write updated data back to JSON file
    with open("DownloadTaskFromSolidity.json", 'w') as file:
        json.dump(data, file, indent=2)

    return returned_tasks


def get_users():
    returned_users = contract.functions.getUsers().call()
    # print(returned_users)
    for user in returned_users:
        user_id = user[0]
        user_isFree = user[1]
        user_ability = user[2]
        user_contact = user[3]
        user_userAddress = user[4]
        task_rating = user[5]
        print(
            f"User ID: {user_id}, user is free: {user_isFree},user ability: {user_ability}, user contact: {user_contact},user address: {user_userAddress}, task rating: {task_rating}")

    # Save these tasks in local machine
    with open("DownloadAgentFromSolidity.json", 'r') as file:
        data = json.load(file)
        print('print data json')
        print(data)

    if isinstance(data, list) and len(data) > 0:
        # Get the parameter structure of the first user
        parameters = data[0]['function']['parameters']
        if 'properties' in parameters:
            properties = parameters['properties']

            # Add each user to JSON file
            for user in returned_users:
                new_user = {
                    'User_id': {'value': user[0], 'type': 'integer'},
                    'is_free': {'value': user[1], 'type': 'bool'},
                    'ability': {'value': user[2], 'type': 'string'},
                    'contact': {'value': user[3], 'type': 'string'},
                    'address': {'value': user[4], 'type': 'string'},
                    'rating': {'value': user[5], 'type': 'integer'}
                }

                # Verify that the user meets the requirements
                if validate_user(new_user, properties):
                    data.append({'type': 'User', 'function': {'name': 'AgentRegisterName', 'description': 'The name of user',
                                                              'parameters': {'type': 'object', 'properties': new_user,
                                                                             'required': ['ability', 'contact']}}})

    # Write updated data back to JSON file
    with open("DownloadAgentFromSolidity.json", 'w') as file:
        json.dump(data, file, indent=2)


    return returned_users

def get_contact(address):
    returned_results = contract.functions.getUserByAddress(address).call()
    return returned_results[3]  # Return the user's contact

def get_ability(address):
    returned_results = contract.functions.getUserByAddress(address).call()
    return returned_results[2]  # Return the user's ability



get_user_by_address_schema = {
    "type": "object",
    "properties": {
        "address": {"type": "string"},
    },
    "required": ["address"]
}
def get_user_by_address(params):
    jsonschema.validate(params, get_user_by_address_schema)

    # Destructuring parameters
    address = params["address"]

    returned_users = contract.functions.getUserByAddress(address).call()
    print(returned_users)
    user_id = returned_users[0]
    user_isFree = returned_users[1]
    user_ability = returned_users[2]
    user_contact = returned_users[3]
    user_address = returned_users[4]
    task_rating = returned_users[5]

    print(
        f"User ID: {user_id }, user isFree: {user_isFree},user ability: {user_ability},user_contact: {user_contact},user_address: {user_address}, task_rating: {task_rating}")
    return returned_users