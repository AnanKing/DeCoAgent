// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AgentCollaborationsContract {
    struct User {
        uint256 id; // User ID
        bool isFree; // Whether the user is free.
        string ability; // User ability
        string contact; // User contact, e.g. email
        address userAddress; // User address
        uint256 rating; //User ratings, which can be expanded to build reputation levels in the future.
    }

    struct Task {
        uint256 id; // task ID
        string requirement; // task requirement
        address owner; // Task owner
        address worker; // Task worker
        bool isDeployed; // Whether the task is deployed
        uint256 price; // Task price
        bool isPay;
        // string result; // The task results can be sent through the contact of the agent (e.g. email), so omit it in the contract.
    }

    mapping(uint256 => User) public users;

    mapping(uint256 => Task) public tasks;

    uint256 public userCount;

    uint256 public taskCount;

    event UserRegistered(
        uint256 id,
        bool isFree,
        string ability,
        string contact,
        address userAddress,
        uint256 rating
    );

    event TaskPosted(
        uint256 id,
        string requirement,
        address owner,
        uint256 price
    );

    event TaskMatched(uint256 id, address worker);

    event TaskDeployed(uint256 id, address worker);

    event WorkerRated(uint256 id, uint256 rating);

    event TaskCompleted(uint256 indexed id, string paymentHashResult);

    event TaskRequested(uint256 indexed userId, uint256 indexed taskId);

    function register(string memory _ability, string memory _contact) public {
        userCount++;

        User memory newUser = User(
            userCount,
            true,
            _ability,
            _contact,
            msg.sender,
            0
        );

        users[userCount] = newUser;

        emit UserRegistered(userCount, true, _ability, _contact, msg.sender, 0);
    }

    function getTaskWithID(uint256 _id)
        public
        view
        returns (
            uint256 id,
            string memory requirement,
            address owner,
            address worker,
            bool isdeployed,
            uint256 price,
            bool isPay
        )
    {
        Task storage task = tasks[_id];

        return (
            task.id,
            task.requirement,
            task.owner,
            task.worker,
            task.isDeployed,
            task.price,
            task.isPay
        );
    }

    function getUserByAddress(address _userAddress)
        public
        view
        returns (
            uint256 id,
            bool isFree,
            string memory ability,
            string memory contact,
            address userAddress,
            uint256 rating
        )
    {
        for (uint256 i = 1; i <= userCount; i++) {
            if (users[i].userAddress == _userAddress) {
                return (
                    users[i].id,
                    users[i].isFree,
                    users[i].ability,
                    users[i].contact,
                    users[i].userAddress,
                    users[i].rating
                );
            }
        }

        revert("User not found");
    }

    function getUserById(uint256 _userId)
        public
        view
        returns (
            uint256 id,
            bool isFree,
            string memory ability,
            string memory contact,
            address userAddress,
            uint256 rating
        )
    {
        require(_userId <= userCount, "User not found");

        User memory user = users[_userId];

        return (
            user.id,
            user.isFree,
            user.ability,
            user.contact,
            user.userAddress,
            user.rating
        );
    }

    function getUsers() public view returns (User[] memory) {
        User[] memory allUser = new User[](userCount);
        for (uint256 i = 1; i <= userCount; i++) {
            allUser[i - 1] = users[i];
        }
        return allUser;
    }

    function getTasks() public view returns (Task[] memory) {
        Task[] memory resTask = new Task[](taskCount);

        for (uint256 i = 1; i <= taskCount; i++) {
            resTask[i - 1] = tasks[i];
        }

        return resTask;
    }

    function queryByAbility(string memory _ability)
        public
        view
        returns (User[] memory)
    {
        User[] memory matchedUsers = new User[](userCount);

        uint256 matchedCount = 0;

        for (uint256 i = 1; i <= userCount; i++) {
            if (
                keccak256(abi.encodePacked(users[i].ability)) ==
                keccak256(abi.encodePacked(_ability)) &&
                users[i].isFree
            ) {
                matchedUsers[matchedCount] = users[i];
                matchedCount++;
            }
        }

        assembly {
            mstore(matchedUsers, matchedCount)
        }

        return matchedUsers;
    }

    function getUserIndexByAddress(address _userAddress)
        internal
        view
        returns (uint256)
    {
        for (uint256 i = 1; i <= userCount; i++) {
            if (users[i].userAddress == _userAddress) {
                return i;
            }
        }
        revert("User not found");
    }

    function postTask(string memory _requirement, uint256 _price) public {
        taskCount++;

        Task memory newTask = Task(
            taskCount,
            _requirement,
            msg.sender,
            address(0),
            false,
            _price,
            false
            //""
        );

        tasks[taskCount] = newTask;

        emit TaskPosted(taskCount, _requirement, msg.sender, _price);
    }

    function deployTask(uint256 _id, address worker) public {
        Task storage task = tasks[_id];

        require(
            task.owner == msg.sender,
            "Only the task owner can deploy the task"
        );

        require(!task.isDeployed, "The task is deployed successfully");
        require(
            users[getUserIndexByAddress(worker)].isFree,
            "Worker must be free to deploy the task"
        );

        if (task.worker == address(0)) {
            tasks[_id].worker = worker;
        }

        emit TaskMatched(_id, task.worker);

        tasks[_id].isDeployed = true;
        users[getUserIndexByAddress(worker)].isFree = false;

        emit TaskDeployed(_id, task.worker);
    }

    function ratingWorker(uint256 _id, uint256 _rating) public {
        Task storage task = tasks[_id];
        require(
            task.owner == msg.sender,
            "Only the owner of task can rate the task"
        );
        require(
            task.isDeployed = true,
            "The taks is not deployed, cannot rating it"
        );
        require(task.isPay = true, "The taks is not pay, cannot rating it");
        for (uint256 i = 1; i <= userCount; i++) {
            if (users[i].userAddress == task.worker) {
                users[i].rating = users[i].rating + _rating;
            }
        }
        emit WorkerRated(_id, _rating);
    }

    function requestTask(uint256 _userId, uint256 _taskId) public {
        require(
            users[_userId].userAddress == msg.sender,
            "Only the user can request a task for themselves"
        );

        Task storage task = tasks[_taskId];

        require(!task.isDeployed, "Task is already deployed");

        users[_userId].isFree = false;

        tasks[_taskId].isDeployed = true;

        emit TaskRequested(_userId, _taskId);
    }

    function completeTask(uint256 _id, string memory _paymentHashResult)
        public
    {
        Task storage task = tasks[_id];

        require(
            task.owner == msg.sender,
            "Only the task owner can complete the task"
        );
        require(task.isDeployed, "The task must be deployed before completion");

        // Assuming you have a suitable method to verify payment results, the authenticity check is omitted here.

        users[getUserIndexByAddress(task.worker)].isFree = true;

        tasks[_id].isPay = true;

        emit TaskCompleted(_id, _paymentHashResult);
    }
}
