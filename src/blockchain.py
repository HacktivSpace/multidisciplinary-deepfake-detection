import hashlib
import json
import time
from typing import List, Dict

class Block:
    def __init__(self, index: int, previous_hash: str, timestamp: float, data: Dict, nonce: int = 0):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.nonce = nonce
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        block_string = f"{self.index}{self.previous_hash}{self.timestamp}{json.dumps(self.data)}{self.nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()

    def __str__(self) -> str:
        return json.dumps(self.__dict__, indent=4)

class Blockchain:
    def __init__(self, difficulty: int = 4):
        self.chain: List[Block] = []
        self.difficulty = difficulty
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = Block(0, "0", time.time(), {"message": "Genesis Block"})
        self.chain.append(genesis_block)

    def get_latest_block(self) -> Block:
        return self.chain[-1]

    def add_block(self, data: Dict):
        latest_block = self.get_latest_block()
        new_block = Block(
            index=latest_block.index + 1,
            previous_hash=latest_block.hash,
            timestamp=time.time(),
            data=data
        )
        self.mine_block(new_block)
        self.chain.append(new_block)

    def mine_block(self, block: Block):
        print(f"Mining block {block.index}...")
        while block.hash[:self.difficulty] != '0' * self.difficulty:
            block.nonce += 1
            block.hash = block.calculate_hash()
        print(f"Block {block.index} mined: {block.hash}")

    def is_chain_valid(self) -> bool:
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block.hash != current_block.calculate_hash():
                print(f"Invalid hash at block {current_block.index}")
                return False

            if current_block.previous_hash != previous_block.hash:
                print(f"Invalid previous hash at block {current_block.index}")
                return False

        return True

    def __str__(self) -> str:
        chain_data = [str(block) for block in self.chain]
        return json.dumps(chain_data, indent=4)

if __name__ == "__main__":
    blockchain = Blockchain(difficulty=4)

    blockchain.add_block({"transaction": "Alice pays Bob 10 BTC"})
    blockchain.add_block({"transaction": "Bob pays Charlie 5 BTC"})
    blockchain.add_block({"transaction": "Charlie pays Dave 2 BTC"})

    print(blockchain)
    print("Blockchain valid:", blockchain.is_chain_valid())
