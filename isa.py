from enum import Enum

MEMORY_SIZE = 2048
STACK_SIZE = 1024
MAX_NUMBER = 1 << 31 - 1
MIN_NUMBER = -(1 << 31)
INT1_ADDRESS = 1
INPUT_PORT_ADDRESS = 2
OUTPUT_PORT_ADDRESS = 3


class Variable:
    def __init__(self, name: str, address: int, data: list[int], is_string: bool):
        self.name = name
        self.address = address
        self.data = data
        self.is_string = is_string


class Opcode(Enum):
    NOP = ("nop", "00000")
    ADD = ("add", "00001")
    SUB = ("sub", "00010")
    MUL = ("mul", "00011")
    DIV = ("div", "00100")
    MOD = ("mod", "00101")
    INC = ("inc", "00110")
    DEC = ("dec", "00111")
    DUP = ("dup", "01000")
    OVER = ("over", "01001")
    SWITCH = ("switch", "01010")
    CMP = ("cmp", "01011")
    JMP = ("jmp", "01100")
    JZ = ("jz", "01101")
    JNZ = ("jnz", "01110")
    CALL = ("call", "01111")
    RET = ("ret", "10000")
    LIT = ("lit", "10001")
    PUSH = ("push", "10010")
    POP = ("pop", "10011")
    DROP = ("drop", "10100")
    EI = ("ei", "10101")
    DI = ("di", "10110")
    IRET = ("iret", "10111")
    HALT = ("halt", "11000")

    def __init__(self, mnemonic: str, binary: str):
        self.mnemonic = mnemonic
        self.binary = binary

    @classmethod
    def from_string(cls, value):
        for opcode in cls:
            if opcode.mnemonic == value:
                return opcode
        raise ValueError(f"{value} is unknown command")

    @classmethod
    def from_binary(cls, value):
        for opcode in cls:
            if opcode.binary == value:
                return opcode
        raise ValueError(f"{value} is unknown command")


class Command:
    def __init__(self, opcode: Opcode, operand=None):
        self.opcode = opcode
        self.operand = operand


def write_commented_code(filename: str, commented_code: str):
    with open(filename, mode="w", encoding="utf-8") as f:
        f.write(commented_code)


def write_code(filename: str, code: str):
    with open(filename, mode="bw") as f:
        f.write(code.encode("utf-8"))


def read_code(filename: str) -> list[int]:
    with open(filename, mode="rb") as f:
        code = f.read()
        code = str(code, encoding="utf-8").splitlines()
        code = map(binary32_to_int, code)
        return list(code)


def value_to_binary32(value: int) -> str:
    return format(value & 0xFFFFFFFF, "032b")


def command_to_binary32(command: Command) -> str:
    return f"{command.opcode.binary}" + "0" * 27


def binary32_to_int(value: str) -> int:
    num = int(value, 2)
    if value[0] == "1":
        num -= 1 << 32
    return num



def int_to_opcode(value: int) -> Opcode:
    value = value_to_binary32(value)[:5]
    return Opcode.from_binary(value)
