import logging
import sys

from isa import (
    INPUT_PORT_ADDRESS,
    MAX_NUMBER,
    MEMORY_SIZE,
    MIN_NUMBER,
    OUTPUT_PORT_ADDRESS,
    STACK_SIZE,
    Opcode,
    int_to_opcode,
    read_code,
)

INSTRUCTION_LIMIT = 1500

ALU_OPCODE_BINARY_HANDLERS = {
    Opcode.ADD: lambda left, right: int(left + right),
    Opcode.SUB: lambda left, right: int(left - right),
    Opcode.MUL: lambda left, right: int(left * right),
    Opcode.DIV: lambda left, right: int(left / right),
    Opcode.MOD: lambda left, right: int(left % right),
    Opcode.CMP: lambda left, right: int(left - right),
}

ALU_OPCODE_SINGLE_HANDLERS = {
    Opcode.INC: lambda left: left + 1,
    Opcode.DEC: lambda left: left - 1,
}


class InterruptionController:
    interruption: bool = None
    interruption_number: int = None

    def __init__(self):
        self.interruption = False
        self.interruption_number = 0

    def generate_interruption(self, number: int) -> None:
        assert number == 1, f"Interruption controller doesn't invoke interruption-{number}"
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
    data_stack = None

    data_stack_top_1 = None

    data_stack_top_2 = None

    data_stack_size = None

    address_stack = None

    address_stack_top = None

    address_stack_size = None

    pc = None

    memory = None

    memory_size = None

    alu: Alu = None

    input_buffer = None

    output_buffer = None

    interruption_controller: InterruptionController = None

    def __init__(self, memory):
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
        assert len(self.data_stack) != self.data_stack_size - 1, "Data stack is overflowed"
        self.data_stack.append(value)

    def signal_read_data_stack(self) -> int:
        assert len(self.data_stack) != 0, "Data stack is empty"
        return self.data_stack.pop()

    def signal_latch_address_stack_top(self, value: int) -> None:
        self.address_stack_top = value

    def signal_write_address_stack(self, value: int) -> None:
        assert len(self.address_stack) != self.address_stack_size - 1, "Address stack is overflowed"
        self.address_stack.append(value)

    def signal_read_address_stack(self) -> int:
        assert len(self.address_stack) != 0, "Address stack is empty"
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
            assert address < self.memory_size, f"Memory doesn't have cell with index {address}"
            self.memory[address] = value

    def signal_read_memory(self, address: int) -> int:
        if address == INPUT_PORT_ADDRESS:
            logging.debug("input: %s", repr(chr(self.input_buffer)))
            return self.input_buffer
        assert address < self.memory_size, f"Memory doesn't have cell with index {address}"
        return self.memory[address]


class ControlUnit:
    tick_counter: int = None

    interruption_enabled: bool = None

    handling_interruption: bool = None

    data_path: DataPath = None

    current_instruction: Opcode = None

    current_operand: int = None

    instruction_executors = None

    def __init__(self, data_path: DataPath):
        self.tick_counter = 0
        self.interruption_enabled = False
        self.handling_interruption = False
        self.data_path = data_path
        self.instruction_executors = {
            Opcode.ADD: self.execute_binary_math_instruction,
            Opcode.SUB: self.execute_binary_math_instruction,
            Opcode.MUL: self.execute_binary_math_instruction,
            Opcode.DIV: self.execute_binary_math_instruction,
            Opcode.MOD: self.execute_binary_math_instruction,
            Opcode.INC: self.execute_unary_math_instruction,
            Opcode.DEC: self.execute_binary_math_instruction,
            Opcode.DUP: self.execute_dup,
            Opcode.OVER: self.execute_over,
            Opcode.SWITCH: self.execute_switch,
            Opcode.CMP: self.execute_cmp,
            Opcode.LIT: self.execute_lit,
            Opcode.PUSH: self.execute_push,
            Opcode.POP: self.execute_pop,
            Opcode.DROP: self.execute_drop,
            Opcode.EI: self.execute_ei,
            Opcode.DI: self.execute_di,
            Opcode.IRET: self.execute_iret,
        }

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

    def decode_and_execute_control_flow_instruction(self, opcode):
        if opcode == Opcode.HALT:
            self.execute_halt()
        if opcode == Opcode.JMP:
            return self.execute_jmp()
        if opcode == Opcode.JZ:
            return self.execute_jz()
        if opcode == Opcode.JNZ:
            return self.execute_jnz()
        if opcode == Opcode.CALL:
            return self.execute_call()
        if opcode == Opcode.RET:
            return self.execute_ret()
        return False

    def execute_halt(self):
        logging.debug("%s", self.__repr__())
        raise StopIteration()

    def execute_jmp(self):
        address = self.data_path.signal_read_memory(self.data_path.pc)
        self.data_path.signal_latch_data_stack_top_1(address)
        self.tick()

        self.data_path.pc = address
        self.tick()

        self.current_operand = address
        logging.debug("%s", self.__repr__())
        return True

    def execute_jz(self):
        if self.data_path.alu.z_flag:
            address = self.data_path.signal_read_memory(self.data_path.pc)
            self.data_path.signal_latch_data_stack_top_1(address)
            self.tick()

            self.data_path.pc = address
            self.tick()

            self.current_operand = address
            logging.debug("%s", self.__repr__())
            return True

        self.data_path.signal_latch_pc(self.data_path.pc + 1)
        self.tick()

        self.current_operand = self.data_path.memory[self.data_path.pc - 1]
        logging.debug("%s", self.__repr__())
        return True

    def execute_jnz(self):
        if not self.data_path.alu.z_flag:
            address = self.data_path.signal_read_memory(self.data_path.pc)
            self.data_path.signal_latch_data_stack_top_1(address)
            self.tick()

            self.data_path.pc = address
            self.tick()

            self.current_operand = address
            logging.debug("%s", self.__repr__())
            return True

        self.data_path.signal_latch_pc(self.data_path.pc + 1)
        self.tick()

        self.current_operand = self.data_path.memory[self.data_path.pc - 1]
        logging.debug("%s", self.__repr__())
        return True

    def execute_call(self):
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

    def execute_ret(self):
        self.data_path.signal_latch_address_stack_top(self.data_path.signal_read_address_stack())
        self.tick()

        self.data_path.signal_latch_pc(self.data_path.address_stack_top)
        self.tick()

        logging.debug("%s", self.__repr__())
        return True

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

        instruction_executor = self.instruction_executors[opcode]
        instruction_executor(opcode)

    def execute_binary_math_instruction(self, opcode):
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

    def execute_unary_math_instruction(self, opcode):
        operand = self.data_path.signal_read_data_stack()
        self.data_path.signal_latch_data_stack_top_1(operand)
        self.tick()

        result = self.data_path.alu.perform(self.data_path.data_stack_top_1, self.data_path.data_stack_top_2, opcode)
        self.data_path.signal_latch_data_stack_top_1(result)
        self.tick()

        self.data_path.signal_write_data_stack(self.data_path.data_stack_top_1)
        self.tick()

        logging.debug("%s", self.__repr__())

    def execute_dup(self, opcode):
        operand = self.data_path.signal_read_data_stack()
        self.data_path.signal_latch_data_stack_top_1(operand)
        self.tick()

        self.data_path.signal_write_data_stack(self.data_path.data_stack_top_1)
        self.tick()

        self.data_path.signal_write_data_stack(self.data_path.data_stack_top_1)
        self.tick()

        logging.debug("%s", self.__repr__())

    def execute_over(self, opcode):
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

    def execute_switch(self, opcode):
        self.data_path.signal_latch_data_stack_top_1(self.data_path.signal_read_data_stack())
        self.tick()

        self.data_path.signal_latch_data_stack_top_2(self.data_path.signal_read_data_stack())
        self.tick()

        self.data_path.signal_write_data_stack(self.data_path.data_stack_top_1)
        self.tick()

        self.data_path.signal_write_data_stack(self.data_path.data_stack_top_2)
        self.tick()

        logging.debug("%s", self.__repr__())

    def execute_cmp(self, opcode):
        operand1 = self.data_path.signal_read_data_stack()
        self.data_path.signal_latch_data_stack_top_1(operand1)
        self.tick()

        operand2 = self.data_path.signal_read_data_stack()
        self.data_path.signal_latch_data_stack_top_2(operand2)
        self.tick()

        self.data_path.alu.perform(self.data_path.data_stack_top_1, self.data_path.data_stack_top_2, Opcode.CMP)
        self.data_path.signal_write_data_stack(self.data_path.data_stack_top_2)
        self.tick()

        self.data_path.signal_write_data_stack(self.data_path.data_stack_top_1)
        self.tick()

        logging.debug("%s", self.__repr__())

    def execute_lit(self, opcode):
        self.data_path.signal_latch_data_stack_top_1(self.data_path.signal_read_memory(self.data_path.pc))
        self.tick()

        self.data_path.signal_write_data_stack(self.data_path.data_stack_top_1)
        self.data_path.signal_latch_pc(self.data_path.pc + 1)
        self.tick()

        self.current_operand = self.data_path.data_stack[-1]
        logging.debug("%s", self.__repr__())

    def execute_push(self, opcode):
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

    def execute_pop(self, opcode):
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

    def execute_drop(self, opcode):
        self.data_path.signal_latch_data_stack_top_1(self.data_path.signal_read_data_stack())
        self.tick()

        logging.debug("%s", self.__repr__())

    def execute_ei(self, opcode):
        self.interruption_enabled = True
        self.tick()

        logging.debug("%s", self.__repr__())

    def execute_di(self, opcode):
        self.interruption_enabled = False
        self.tick()

        logging.debug("%s", self.__repr__())

    def execute_iret(self, opcode):
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

    def __repr__(self) -> str:
        registers_repr = "TICK: {:10} PC: {:10} TODS1: {:10} TODS2: {:10} TOAS: {:10} Z_FLAG: {:1}".format(
            str(self.tick_counter),
            str(self.data_path.pc),
            str(self.data_path.data_stack_top_1),
            str(self.data_path.data_stack_top_2),
            str(self.data_path.address_stack_top),
            int(self.data_path.alu.z_flag),
        )

        data_stack_repr = "DATA_STACK: {}".format(self.data_path.data_stack)
        address_stack_repr = "ADDRESS_STACK: {}".format(self.data_path.address_stack)

        instruction_repr = self.current_instruction.mnemonic

        if self.current_operand is not None:
            instruction_repr += " {}".format(self.current_operand)

        return "{} \t{}\n\t   {}\n\t   {}".format(registers_repr, instruction_repr, data_stack_repr, address_stack_repr)


def initiate_interruption(control_unit, input_tokens):
    if len(input_tokens) != 0:
        next_token = input_tokens[0]
        if control_unit.tick_counter >= next_token[0]:
            control_unit.data_path.interruption_controller.generate_interruption(1)
            if next_token[1]:
                control_unit.data_path.input_buffer = ord(next_token[1])
            else:
                control_unit.data_path.input_buffer = 0
            return input_tokens[1:]
    return input_tokens


def simulation(code, input_tokens):
    data_path = DataPath(code)
    control_unit = ControlUnit(data_path)

    control_unit.initialization_cycle()

    instruction_counter = 0
    try:
        while instruction_counter < INSTRUCTION_LIMIT:
            input_tokens = initiate_interruption(control_unit, input_tokens)
            control_unit.check_and_handle_interruption()
            control_unit.decode_and_execute_instruction()
            instruction_counter += 1
    except StopIteration:
        pass

    if instruction_counter == INSTRUCTION_LIMIT:
        logging.warning("Instruction limit reached")

    return data_path.output_buffer, instruction_counter, control_unit.tick_counter


def main(code_file: str, input_file: str):
    code = read_code(code_file)
    with open(input_file, encoding="utf-8") as f:
        input_text = f.read().strip()
        if not input_text:
            input_tokens = []
        else:
            input_tokens = eval(input_text)

    output, instruction_counter, ticks = simulation(code, input_tokens)

    print("".join(output) + "\n")
    print(f"instr_counter: {instruction_counter} ticks: {ticks}")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    assert len(sys.argv) == 3, "Invalid usage: python3 machine.py <code_file> <input_file>"
    _, code_file, input_file = sys.argv
    main(code_file, input_file)
