import CallACPFunctions

def AgentRegisteredFunction(ability,contact):
    print("Start Executing AgentRegisteredFunction")
    params = {
        "ability": ability,
        "contact": contact,
    }
    CallACPFunctions.register(params)


def QueryAbilityFunction(ability):
    print("Start Executing QueryAbilityFunction")
    params = {
        "ability": ability,
    }
    CallACPFunctions.query_by_ability(params)

def getTaskWithIDFunction(id):
    print("Start Executing getTaskWithIDFunction")
    params = {
        "id": id,
    }
    CallACPFunctions.get_task_with_ID(params)

def getTasksFunction():
    print("Start Executing getTasksFunction")
    CallACPFunctions.get_tasks()

def getUsersFunction():
    print("Start Executing getUsersFunction")
    CallACPFunctions.get_users()

def getUserByAddressFunction(address):
    print("Start Executing getUserByAddressFunction")
    params = {
        "address": address,
    }
    CallACPFunctions.get_user_by_address(params)


def PostTaskFunction(requirement,price):
    print("Start Executing PostTaskFunction")
    params = {
        "requirement": requirement,
        "price": price,
    }
    CallACPFunctions.post_task(params)


def DeployTaskFunction(id):
    print("Start Executing DeployTaskFunction")
    params = {
        "id": id,
    }
    CallACPFunctions.deploy_task(params)

def RequestTaskFunction(id):
    print("Start Executing RequestTaskFunction")
    params = {
        "id": id,

    }
    CallACPFunctions.request_task(params)


def RatingWorkerFunction(id,rating):
    print("StartExecuting RatingWorkerFunction")
    params = {
        "id": id,
        "rating": rating,
    }
    CallACPFunctions.rating_worker(params)


