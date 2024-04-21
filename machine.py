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
    int_to_opcode
)

INSTRUCTION_LIMIT = 10000


ALU_OPCODE_BINARY_HANDLERS: dict[Opcode, Callable[[int, int], int]] = {
    Opcode.ADD: lambda left, right: int((left + right)),
    Opcode.SUB: lambda left, right: int((left - right)),
    Opcode.MUL: lambda left, right: int((left * right)),
    Opcode.DIV: lambda left, right: int((left / right)), 
    Opcode.MOD: lambda left, right: int((left % right)),
    Opcode.CMP: lambda left, right: int((left - right))
}


ALU_OPCODE_SINGLE_HANDLERS: dict[Opcode, Callable[[int], int]] = {
    Opcode.INC: lambda left: left + 1,
    Opcode.DEC: lambda left: left - 1
}

class InterruptionController:
    interruption: bool = None
    interruption_number: int = None

    def __init__(self):
        self.interruption = False
        self.interruption_number = 0

    def generate_interruption(self, number: int) -> None:
        assert (
            number == 1
        ), f"Interruption controller doesn't invoke interruption-{number}"
        self.interruption = True
        self.interruption_number = number

class Alu:

    z_flag = None

    def __init__(self):
        self.z_flag = 0

    def perform(self, left: int, right: int, opcode: Opcode) -> int:
        assert (
            opcode in ALU_OPCODE_BINARY_HANDLERS or opcode in ALU_OPCODE_SINGLE_HANDLERS
        ), f"Unknown ALU command {opcode.mnemonic}"
        if opcode in ALU_OPCODE_BINARY_HANDLERS:
            handler = ALU_OPCODE_BINARY_HANDLERS[opcode]
            value = handler(left, right)
        else:
            handler = ALU_OPCODE_SINGLE_HANDLERS[opcode]
            value = handler(left)
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

    input_buffer: int = None

    output_buffer: list[str] = None

    interruption_controller: InterruptionController = None

    def __init__(self, memory: list[int]):
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

        self.input_buffer = 0
        self.output_buffer = []
        self.interruption_controller = InterruptionController()

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
            character = chr(value)
            logging.debug("output: %s << %s", repr("".join(self.output_buffer)), repr(character))
            self.output_buffer.append(character)
        else: 
            assert (
                address < self.memory_size
            ), f"Memory doesn't have cell with index {address}"
            self.memory[address] = value

    def signal_read_memory(self, address: int) -> int:
        if address == INPUT_PORT_ADDRESS:
            logging.debug("input: %s", repr(chr(self.input_buffer)))
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

    interruption_enabled: bool = None

    handling_interruption: bool = None
    
    data_path: DataPath = None

    current_instruction: Opcode = None

    current_operand: int = None

    def __init__(self, data_path: DataPath):
        self.tick_counter = 0
        self.interruption_enabled = False
        self.handling_interruption = False
        self.data_path = data_path
    
    def tick(self):
        self.tick_counter += 1

    def initialization_cycle(self):
        start_address = self.data_path.signal_read_memory(self.data_path.pc)
        self.data_path.signal_latch_data_stack_top_1(start_address)
        self.tick()
        self.data_path.signal_latch_pc(self.data_path.data_stack_top_1)
        self.tick()

    def check_and_handle_interruption(self) -> None:
        if not self.interruption_enabled:
            return
        if not self.data_path.interruption_controller.interruption:
            return
        if self.handling_interruption:
            return
        
        self.handling_interruption = True
        self.data_path.signal_latch_address_stack_top(self.data_path.pc)
        self.tick()

        self.data_path.signal_write_address_stack(self.data_path.address_stack_top)
        self.data_path.signal_latch_pc(self.data_path.interruption_controller.interruption_number)
        self.tick()

        address = self.data_path.signal_read_memory(self.data_path.pc)
        self.data_path.signal_latch_data_stack_top_1(address)
        self.tick()

        self.data_path.signal_latch_pc(self.data_path.data_stack_top_1)
        self.tick()

        logging.debug("START HANDLING INTERRUPTION")
        return

    def decode_and_execute_control_flow_instruction(self, opcode: Opcode) -> bool: 
        if opcode == Opcode.HALT:
            logging.debug("%s", self.__repr__())
            raise StopIteration()
        if opcode == Opcode.JMP:
            address = self.data_path.signal_read_memory(self.data_path.pc)
            self.data_path.signal_latch_data_stack_top_1(address)
            self.tick()

            self.data_path.pc = address
            self.tick()

            self.current_operand = address
            logging.debug("%s", self.__repr__())
            return True
        if opcode == Opcode.JZ:
            if self.data_path.alu.z_flag:
                address = self.data_path.signal_read_memory(self.data_path.pc)
                self.data_path.signal_latch_data_stack_top_1(address)
                self.tick()

                self.data_path.pc = address
                self.tick()

                self.current_operand = address
                logging.debug("%s", self.__repr__())
                return True
            else:
                self.data_path.signal_latch_pc(self.data_path.pc + 1)
                self.tick()
                
                self.current_operand = self.data_path.memory[self.data_path.pc - 1]
                logging.debug("%s", self.__repr__())
                return True
        if opcode == Opcode.JNZ:
            if not self.data_path.alu.z_flag:
                address = self.data_path.signal_read_memory(self.data_path.pc)
                self.data_path.signal_latch_data_stack_top_1(address)
                self.tick()

                self.data_path.pc = address
                self.tick()

                self.current_operand = address
                logging.debug("%s", self.__repr__())
                return True
            else:
                self.data_path.signal_latch_pc(self.data_path.pc + 1)
                self.tick()
                
                self.current_operand = self.data_path.memory[self.data_path.pc - 1]
                logging.debug("%s", self.__repr__())
                return True
        if opcode == Opcode.CALL:
            address = self.data_path.signal_read_memory(self.data_path.pc)
            self.data_path.signal_latch_data_stack_top_1(address)
            self.tick()

            self.data_path.signal_latch_pc(self.data_path.pc + 1)
            self.tick()

            self.data_path.signal_latch_address_stack_top(self.data_path.pc)
            self.tick()

            self.data_path.signal_latch_pc(self.data_path.data_stack_top_1)
            self.data_path.signal_write_address_stack(self.data_path.address_stack_top)
            self.tick()

            self.current_operand = address
            logging.debug("%s", self.__repr__())
            return True
        if opcode == Opcode.RET:
            self.data_path.signal_latch_address_stack_top(self.data_path.signal_read_address_stack())
            self.tick()

            self.data_path.signal_latch_pc(self.data_path.address_stack_top)
            self.tick()

            logging.debug("%s", self.__repr__())
            return True

        return False

    def decode_and_execute_instruction(self):
        instruction = self.data_path.signal_read_memory(self.data_path.pc)
        self.tick()
        self.data_path.signal_latch_pc(self.data_path.pc + 1)
        self.tick()

        opcode = int_to_opcode(instruction)
        self.current_instruction = opcode
        self.current_operand = None
        if self.decode_and_execute_control_flow_instruction(opcode):
            return
        
        if opcode == Opcode.NOP:
            logging.debug("%s", self.__repr__())
            return
        if opcode in [Opcode.ADD, Opcode.SUB, Opcode.MUL, Opcode.DIV, Opcode.MOD]:
            operand1 = self.data_path.signal_read_data_stack()
            self.data_path.signal_latch_data_stack_top_1(operand1)
            self.tick()

            operand2 = self.data_path.signal_read_data_stack()
            self.data_path.signal_latch_data_stack_top_2(operand2)
            self.tick()

            result = self.data_path.alu.perform(self.data_path.data_stack_top_1, self.data_path.data_stack_top_2, opcode)
            self.data_path.signal_latch_data_stack_top_1(result)
            self.tick()

            self.data_path.signal_write_data_stack(self.data_path.data_stack_top_1)
            self.tick()

            logging.debug("%s", self.__repr__())
            return
        if opcode in [Opcode.INC, Opcode.DEC]:
            operand = self.data_path.signal_read_data_stack()
            self.data_path.signal_latch_data_stack_top_1(operand)
            self.tick()

            result = self.data_path.alu.perform(self.data_path.data_stack_top_1, self.data_path.data_stack_top_2, opcode)
            self.data_path.signal_latch_data_stack_top_1(result)
            self.tick()

            self.data_path.signal_write_data_stack(self.data_path.data_stack_top_1)

            logging.debug("%s", self.__repr__())
            return
        if opcode == Opcode.DUP:
            operand = self.data_path.signal_read_data_stack()
            self.data_path.signal_latch_data_stack_top_1(operand)
            self.tick()

            self.data_path.signal_write_data_stack(self.data_path.data_stack_top_1)
            self.tick

            self.data_path.signal_write_data_stack(self.data_path.data_stack_top_1)
            self.tick

            logging.debug("%s", self.__repr__())    
            return
        if opcode == Opcode.OVER:
            operand1 = self.data_path.signal_read_data_stack()
            self.data_path.signal_latch_data_stack_top_1(operand1)
            self.tick()

            operand2 = self.data_path.signal_read_data_stack()
            self.data_path.signal_latch_data_stack_top_2(operand2)
            self.tick()

            self.data_path.signal_write_data_stack(self.data_path.data_stack_top_2)
            self.tick()

            self.data_path.signal_write_data_stack(self.data_path.data_stack_top_1)
            self.tick()

            self.data_path.signal_write_data_stack(self.data_path.data_stack_top_2)
            self.tick()
            
            logging.debug("%s", self.__repr__())
            return
        if opcode == Opcode.SWITCH:
            self.data_path.signal_latch_data_stack_top_1(self.data_path.signal_read_data_stack())
            self.tick()

            self.data_path.signal_latch_data_stack_top_2(self.data_path.signal_read_data_stack())
            self.tick()

            self.data_path.signal_write_data_stack(self.data_path.data_stack_top_1)
            self.tick()

            self.data_path.signal_write_data_stack(self.data_path.data_stack_top_2)
            self.tick()

            logging.debug("%s", self.__repr__())
            return
        if opcode == Opcode.CMP:
            operand1 = self.data_path.signal_read_data_stack()
            self.data_path.signal_latch_data_stack_top_1(operand1)
            self.tick()

            operand2 = self.data_path.signal_read_data_stack()
            self.data_path.signal_latch_data_stack_top_2(operand2)
            self.tick()

            result = self.data_path.alu.perform(self.data_path.data_stack_top_1, self.data_path.data_stack_top_2, opcode)
            self.data_path.signal_write_data_stack(self.data_path.data_stack_top_2)
            self.tick()

            self.data_path.signal_write_data_stack(self.data_path.data_stack_top_1)
            self.tick()

            logging.debug("%s", self.__repr__())
            return
        if opcode == Opcode.LIT:
            self.data_path.signal_latch_data_stack_top_1(self.data_path.signal_read_memory(self.data_path.pc))
            self.tick()

            self.data_path.signal_write_data_stack(self.data_path.data_stack_top_1)
            self.data_path.signal_latch_pc(self.data_path.pc + 1)
            self.tick()

            self.current_operand = self.data_path.data_stack[-1]
            logging.debug("%s", self.__repr__())
            return
        if opcode == Opcode.PUSH:
            address = self.data_path.signal_read_data_stack()
            self.data_path.signal_latch_data_stack_top_1(address)
            self.data_path.signal_latch_address_stack_top(self.data_path.pc)
            self.tick()

            self.data_path.signal_latch_pc(self.data_path.data_stack_top_1)
            self.tick()

            self.data_path.signal_latch_data_stack_top_1(self.data_path.signal_read_memory(self.data_path.pc))
            self.tick()

            self.data_path.signal_write_data_stack(self.data_path.data_stack_top_1)
            self.data_path.signal_latch_pc(self.data_path.address_stack_top)
            self.tick()

            logging.debug("%s", self.__repr__())
            return            
        if opcode == Opcode.POP:
            address = self.data_path.signal_read_data_stack()
            self.data_path.signal_latch_data_stack_top_1(address)
            self.tick()

            operand = self.data_path.signal_read_data_stack()
            self.data_path.signal_latch_data_stack_top_2(operand)
            self.data_path.signal_latch_address_stack_top(self.data_path.pc)
            self.tick()

            self.data_path.signal_latch_pc(self.data_path.data_stack_top_1)
            self.tick()

            self.data_path.signal_write_memory(self.data_path.pc, self.data_path.data_stack_top_2)
            self.tick()

            self.data_path.signal_latch_pc(self.data_path.address_stack_top)
            self.tick()

            logging.debug("%s", self.__repr__())
            return     
        if opcode == Opcode.DROP:
            self.data_path.signal_latch_data_stack_top_1(self.data_path.signal_read_data_stack())
            self.tick()

            logging.debug("%s", self.__repr__())
            return
        if opcode == Opcode.EI:
            self.interruption_enabled = True
            self.tick()

            logging.debug("%s", self.__repr__())
            return
        if opcode == Opcode.DI:
            self.interruption_enabled = False
            self.tick()

            logging.debug("%s", self.__repr__())
            return
        if opcode == Opcode.IRET:
            if not self.handling_interruption:
                return
             
            address = self.data_path.signal_read_address_stack()
            self.data_path.signal_latch_address_stack_top(address)
            self.tick()

            self.data_path.signal_latch_pc(self.data_path.address_stack_top)
            self.handling_interruption = False
            self.data_path.interruption_controller.interruption = False
            self.tick()

            logging.debug("%s", self.__repr__())
            logging.debug("STOP HANDLING INTERRUPTION")
            return
        
    def __repr__(self) -> str:
        registers_repr = "TICK: {:10} PC: {:10} TODS1: {:10} TODS2: {:10} TOAS: {:10} Z_FLAG: {:1}".format(
            str(self.tick_counter),
            str(self.data_path.pc),
            str(self.data_path.data_stack_top_1),
            str(self.data_path.data_stack_top_2),
            str(self.data_path.address_stack_top),
            int(self.data_path.alu.z_flag)
        )

        data_stack_repr = "DATA_STACK: {}".format(self.data_path.data_stack)
        address_stack_repr = "ADDRESS_STACK: {}".format(self.data_path.address_stack)

        instruction_repr = self.current_instruction.mnemonic

        if self.current_operand != None:
            instruction_repr += " {}".format(self.current_operand)

        return "{} \t{}\n\t   {}\n\t   {}".format(registers_repr, instruction_repr, data_stack_repr, address_stack_repr)
    

def simulation(code: list[int], input_tokens: list[tuple[str, int]]) -> tuple[list[str], int, int]:
    data_path = DataPath(code)
    control_unit = ControlUnit(data_path)
    
    control_unit.initialization_cycle()

    instruction_counter = 0
    try:
        while instruction_counter < INSTRUCTION_LIMIT:
            print(data_path.memory[:14])
            if len(input_tokens) != 0:
                next_token = input_tokens[0]
                if control_unit.tick_counter >= next_token[0]:
                    data_path.interruption_controller.generate_interruption(1)
                    if next_token[1]:
                        data_path.input_buffer = ord(next_token[1])
                    else:
                        data_path.input_buffer = 0
                    input_tokens = input_tokens[1:]

            control_unit.check_and_handle_interruption()
            control_unit.decode_and_execute_instruction()
            instruction_counter += 1
    except StopIteration:
        pass
    
    return data_path.output_buffer, instruction_counter, control_unit.tick_counter


def main(code_file: str, input_file: str):
    code = read_code(code_file)
    with open(input_file, mode="r", encoding="utf-8") as f:
        input_text = f.read().strip()
        if not input_text:
            input_tokens = []
        else:
            input_tokens = eval(input_text)

    output, instruction_counter, ticks = simulation(code, input_tokens)

    print("output:", "".join(output))
    print("instruction counter:", instruction_counter)
    print("ticks:", ticks)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    assert (
        len(sys.argv) == 3
    ), "Invalid usage: python3 machine.py <code_file> <input_file>"
    _, code_file, input_file = sys.argv
    main(code_file, input_file)
