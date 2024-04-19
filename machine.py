import logging, sys

from typing import Callable

from isa import (
    MAX_NUMBER,
    MIN_NUMBER,
    MEMORY_SIZE,
    STACK_SIZE,
    INPUT_PORT_ADDRESS, 
    OUTPUT_PORT_ADDRESS,
    Opcode,
    read_code, 
    value_to_binary32,
    int_to_opcode
)

INSTRUCTION_LIMIT = 10000


ALU_OPCODE_HANDLERS: dict[Opcode, Callable[[int, int], int]] = {
    Opcode.ADD: lambda left, right: (left + right),
    Opcode.SUB: lambda left, right: (left - right),
    Opcode.MUL: lambda left, right: (left * right),
    Opcode.DIV: lambda left, right: (left / right), 
    Opcode.MOD: lambda left, right: (left % right),
    Opcode.CMP: lambda left, right: (left - right)
}


class Alu:

    z_flag = None

    def __init__(self):
        self.z_flag = 0

    def perform(self, left: int, right: int, opcode: Opcode) -> int:
        assert (
            opcode in ALU_OPCODE_HANDLERS
        ), f"Unknown ALU command {opcode.mnemonic}"
        handler = ALU_OPCODE_HANDLERS[opcode]
        value = handler(left, right)
        value = self.handle_overflow(value)
        self.set_flags(value)
        return value

    def handle_overflow(self, value: int) -> int:
        if value > MAX_NUMBER:
            value %= MAX_NUMBER
        elif value < MIN_NUMBER:
            value %= abs(MIN_NUMBER)
        return value

    def set_flags(self, value) -> None:
        if value == 0:
            self.z_flag = True
        else:
            self.z_flag = False


class DataPath:

    data_stack: list[int] = None

    data_stack_top_1: int = None

    data_stack_top_2: int = None

    data_stack_size: int = None

    address_stack: list[int] = None

    address_stack_top: int = None

    address_stack_size: int = None

    pc: int = None

    memory: list[int] = None

    memory_size: int = None

    alu: Alu = None

    input_tokens: list[tuple[str, int]] = None

    input_buffer: int = None

    output_buffer: int = None

    def __init__(self, memory: list[int], input_tokens: list[tuple[str, int]]):
        self.data_stack = [] 
        self.data_stack_top_1 = 0
        self.data_stack_top_2 = 0
        self.data_stack_size = STACK_SIZE

        self.address_stack = []
        self.address_stack_top = 0
        self.address_stack_size = STACK_SIZE

        self.pc = 0

        self.memory = [0] * MEMORY_SIZE
        for i in range(len(memory)):
            self.memory[i] = memory[i]
        self.memory_size = MEMORY_SIZE

        self.alu = Alu()

        self.input_tokens = input_tokens
        self.input_buffer = 0
        self.output_buffer = 0

    def signal_latch_data_stack_top_1(self, value: int) -> None:
        self.data_stack_top_1 = value

    def signal_latch_data_stack_top_2(self, value: int) -> None:
        self.data_stack_top_2 = value

    def signal_write_data_stack(self, value: int) -> None:
        assert (
            len(self.data_stack) != self.data_stack_size - 1
        ), "Data stack is overflowed"
        self.data_stack.append(value)

    def signal_read_data_stack(self) -> int:
        assert (
            len(self.data_stack) != 0
        ), "Data stack is empty"
        return self.data_stack.pop()
    
    def signal_latch_address_stack_top(self, value: int) -> None:
        self.address_stack_top = value

    def signal_write_address_stack(self, value: int) -> None:
        assert (
            len(self.address_stack) != self.address_stack_size - 1
        ), "Address stack is overflowed"
        self.address_stack.append(value)

    def signal_read_address_stack(self) -> int:
        assert (
            len(self.address_stack) != 0
        ), "Address stack is empty"
        return self.address_stack.pop()
    
    def signal_latch_pc(self, value: int) -> None:
        self.pc = value

    def signal_write_memory(self, address: int, value: int) -> None:
        if address == INPUT_PORT_ADDRESS:
            self.input_buffer = value
        elif address == OUTPUT_PORT_ADDRESS:
            self.output_buffer = value
        else: 
            assert (
                address < self.memory_size
            ), f"Memory doesn't have cell with index {address}"
            self.memory[address] = value

    def signal_read_memory(self, address: int) -> int:
        if address == INPUT_PORT_ADDRESS:
            return self.input_buffer
        elif address == OUTPUT_PORT_ADDRESS:
            return self.output_buffer
        else:
            assert (
                address < self.memory_size
            ), f"Memory doesn't have cell with index {address}"
            return self.memory[address]
        

class ControlUnit:

    tick_counter: int = None
    
    data_path: DataPath = None

    def __init__(self, data_path: DataPath):
        self.tick_counter = 0
        self.data_path = data_path
    
    def tick(self):
        self.tick_counter += 1

    def initialization_cycle(self):
        start_address = self.data_path.signal_read_memory(self.data_path.pc)
        self.data_path.signal_latch_data_stack_top_1(start_address)
        self.tick()
        self.data_path.signal_latch_pc(self.data_path.data_stack_top_1)
        self.tick()

    def decode_and_execute_control_flow_instruction(self, opcode: Opcode) -> bool: 
        if opcode == Opcode.HALT:
            raise StopIteration()

        if opcode == Opcode.JMP:
            address = self.data_path.signal_read_memory(self.data_path.pc)
            self.data_path.signal_latch_data_stack_top_1(address)
            self.tick()
            self.data_path.pc = address
            self.tick()
            return True
        
        if opcode == Opcode.JZ:
            if self.data_path.alu.z_flag:
                address = self.data_path.signal_read_memory(self.data_path.pc)
                self.data_path.signal_latch_data_stack_top_1(address)
                self.tick()
                self.data_path.pc = address
                self.tick()
                return True
            else:
                self.data_path.signal_latch_pc(self.data_path.pc + 1)
                self.tick()

        return False

    def decode_and_execute_instruction(self):
        instruction = self.data_path.signal_read_memory(self.data_path.pc)
        self.tick()
        self.data_path.signal_latch_pc(self.data_path.pc + 1)
        self.tick()

        opcode = int_to_opcode(instruction)
        if self.decode_and_execute_control_flow_instruction(opcode):
            return
        
        if opcode == Opcode.NOP:
            return
        if opcode in [Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV, Opcode.MOD]:
            operand1 = self.data_path.signal_read_data_stack()
            self.data_path.signal_latch_data_stack_top_1(operand1)
            self.tick()

            operand2 = self.data_path.signal_read_data_stack()
            self.data_path.signal_latch_data_stack_top_2(operand2)
            self.tick()

            result = self.data_path.alu.perform(operand1, operand2, opcode)
            self.data_path.signal_latch_data_stack_top_1(result)
            self.tick()

            self.data_path.signal_write_data_stack(self.data_path.data_stack_top_1)
            self.tick()

            return
        if opcode == Opcode.CMP:
            operand1 = self.data_path.signal_read_data_stack()
            self.data_path.signal_latch_data_stack_top_1(operand1)
            self.tick()

            operand2 = self.data_path.signal_read_data_stack()
            self.data_path.signal_latch_data_stack_top_2(operand2)
            self.tick()

            result = self.data_path.alu.perform(operand1, operand2, opcode)
            
            self.data_path.signal_write_data_stack(self.data_path.data_stack_top_2)
            self.tick()

            self.data_path.signal_write_data_stack(self.data_path.data_stack_top_1)
            self.tick()

            return
        if opcode == Opcode.LIT:
            self.data_path.signal_latch_data_stack_top_1(self.data_path.signal_read_memory(self.data_path.pc))
            self.tick()

            self.data_path.signal_write_data_stack(self.data_path.data_stack_top_1)
            self.data_path.signal_latch_pc(self.data_path.pc + 1)
            self.tick()

            return
        if opcode == Opcode.PUSH:
            self.data_path.signal_latch_address_stack_top(self.data_path.pc)
            operand_address = self.data_path.signal_read_memory(self.data_path.pc)
            self.data_path.signal_latch_data_stack_top_1(operand_address)
            self.tick()

            self.data_path.signal_latch_pc(self.data_path.data_stack_top_1)
            self.tick()

            operand = self.data_path.signal_read_memory(self.data_path.pc)
            self.data_path.signal_latch_data_stack_top_1(operand)
            self.tick()

            self.data_path.signal_write_data_stack(self.data_path.data_stack_top_1)
            self.data_path.signal_latch_pc(self.data_path.address_stack_top)
            self.tick()

            self.data_path.signal_latch_pc(self.data_path.pc + 1)
            self.tick()

            return
        if opcode == Opcode.POP:
            self.data_path.signal_latch_data_stack_top_2(self.data_path.signal_read_data_stack())
            address = self.data_path.signal_read_memory(self.data_path.pc)
            self.data_path.signal_latch_data_stack_top_1(address)
            self.tick()

            self.data_path.signal_latch_address_stack_top(self.data_path.pc)
            self.tick()

            self.data_path.signal_latch_pc(self.data_path.data_stack_top_1)
            self.tick()

            self.data_path.signal_write_memory(self.data_path.pc, self.data_path.data_stack_top_2)
            self.tick()

            self.data_path.signal_latch_pc(self.data_path.address_stack_top)
            self.tick()

            self.data_path.signal_latch_pc(self.data_path.pc + 1)
            self.tick()

            return
        if opcode == Opcode.DROP:
            self.data_path.signal_latch_data_stack_top_1(self.data_path.signal_read_data_stack())
            self.tick()


def simulation(code: list[int], input_tokens: list[tuple[str, int]]):
    data_path = DataPath(code, input_tokens)
    control_unit = ControlUnit(data_path)
    
    control_unit.initialization_cycle()

    instruction_counter = 0
    try:
        while instruction_counter < INSTRUCTION_LIMIT:
            control_unit.decode_and_execute_instruction()
            print(data_path.data_stack)
            print(data_path.memory[:30])
            instruction_counter += 1
    except StopIteration:
        pass


def main(code_file: str, input_file: str):
    code = read_code(code_file)
    with open(input_file, mode="r", encoding="utf-8") as f:
        input_text = f.read().strip()
        if not input_text:
            input_tokens = []
        else:
            input_tokens = eval(input_text)
    simulation(code, input_tokens)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    assert (
        len(sys.argv) == 3
    ), "Invalid usage: python3 machine.py <code_file> <input_file>"
    _, code_file, input_file = sys.argv
    main(code_file, input_file)
