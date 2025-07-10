# DecAutoCollaborationPlatformForAgent
Decentralized Autonomous Collaboration Framework for Large Language Model Empowered Agents


Jin, Anan, Yuhang Ye, Brian Lee, and Yuansong Qiao. "DeCoAgent: Large Language Model Empowered Decentralized Autonomous Collaboration Agents Based on Smart Contracts." IEEE Access (2024). https://doi.org/10.1109/ACCESS.2024.3481641


## 1. Introduction Environment
```
Remix version 0.8.17
Ganache version V2.7.1
MteaMask version 11.7.2
Python 3.8
```
## 2. File introduction

```
AgentCollaborationPlatform.sol is a smart contract file.
TestGenerateStepsForACP.py: This file is used to generate the workflow
TestACPwithGPT.py: This file is used to execute each action of the workflow. 
MappingFunctions.py:  This file is used to map the action to functions of the smart contract.
CallACPFunctions.py: This file is used to execute the function of the smart contract.
OwnerSearchMultiWorkers.py: This file is used for the task owner to search for a worker. (task owner=-driven)
WorkerSearchMultiTasks.py: This file is used for workers to search for a task. (worker driven)
```
## 3. Hw to use

```
1. Install all the above environments.  
2. Copy the AgentCollaborationPlatform.sol code to Remix and connect your MetaMask wallet. 
3. Create a workspace in Ganache and connect with your MetaMask wallet. 
4. Compile and run the smart contract code in Remix, you will get a smart contract address and the ABI.
5. Copy the ABI from Remix and paste it in contract_abi.json file.
6. In CallACPFunctions.py file, you should replace these parameters. Copy the ABI file name to replace in 13 lines, and copy the smart contract address to replace in 17 lines. copy the RPC server link from Ganache to replace it in 8 lines, and copy the account address and private key in 23 and 25 lines.  
7. Copy your OpenAI Key in the TestGenerateStepsForACP.py, TestACPwithGPT.py, OwnerSearchMultiWorkers.py, and WorkerSearchMultiTasks.py 
8. Run the TestGenerateStepsForACP.py and test different functions.
```
