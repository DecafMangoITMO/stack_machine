import sys
import re

from isa import (
    MAX_NUMBER,
    MIN_NUMBER,
    MEMORY_SIZE,
    INPUT_PORT_ADDRESS,
    OUTPUT_PORT_ADDRESS,
    Variable,
    Opcode,
    Command,
    write_commented_code,
    write_code,
    value_to_binary32,
    command_to_binary32
)

SECTION_DATA = "section .data:"
SECTION_TEXT = "section .text:"
INTERRUPTION_1 = ".int1"


def remove_comment(line: str) -> str:
    return re.sub(r";.*", "", line)


# Удаление лишних пробелов между словами и пробелов по концам строки
def remove_extra_spaces(line: str) -> str:
    line = line.strip()
    return re.sub(r"\s+", " ", line)


# Очистка исходного текста от комментариев, лишних пробелов и пустых строк
def clean_source(source: str) -> str:
    lines = source.splitlines()

    lines = map(remove_comment, lines)
    lines = map(remove_extra_spaces, lines)
    lines = filter(bool, lines)  # для удаления пустых строк

    return "\n".join(lines)


def is_integer(value: str) -> bool:
    return bool((re.fullmatch(r"^-?\d+$", value)))


def is_string(value: str) -> bool:
    return bool((re.fullmatch(r"^(\".*\")|(\'.*\')$", value)))


# Cтадия обработки секции data - выделение памяти под константы, строки, буфферы и переменные-ссылки
def translate_section_data(section_data: str, address: int, variables: dict[str, Variable]) -> tuple[
    dict[str, Variable], int]:
    lines = section_data.splitlines()
    reference_variables: dict[str, str] = {}

    for line in lines:
        name, value = map(str.strip, line.split(":", 1))

        if is_integer(value):
            value = int(value)
            assert (
                    MIN_NUMBER <= value <= MAX_NUMBER
            ), f"Value {value} is out of bound"
            variable = Variable(name, address, [value], False)
            variables[name] = variable
            address += 1
        elif is_string(value):
            chars = [ord(char) for char in value[1: -1]] + [0]
            variable = Variable(name, address, chars, True)
            variables[name] = variable
            address += len(chars)
        elif value.startswith("bf"):
            _, size = value.split(" ", 1)
            size = int(size)
            variable = Variable(name, address, [0] * size, False)
            variables[name] = variable
            address += size
        else:
            variable = Variable(name, address, [0], False)
            variables[name] = variable
            reference_variables[name] = value
            address += 1

    # Запись адресов в переменные-ссылки
    for reference_variable_name in reference_variables:
        target_variable_name = reference_variables[reference_variable_name]
        variables[reference_variable_name].data = [variables[target_variable_name].address]

    assert (
            address < MEMORY_SIZE
    ), "This programm is too big for proccessor's memory"

    return variables, address


# Первая стадия обработки секции text - выделение меток и их удаление
def translate_section_text_stage_1(section_text: str, address: int) -> tuple[str, dict[str, int]]:
    lines = section_text.splitlines()
    labels: dict[str, int] = {}
    commands = []

    for line in lines:
        if line.startswith("."):
            labels[line[:-1]] = address
        else:
            commands.append(line)
            address += 1
            if len(line.split(" ")) == 2:
                address += 1  # Если команда с операндом, то указатель еще раз смещаем (тк на след. ячейке памяти будет лежать операнд)

    return "\n".join(commands), labels


def translate_command(line: str, labels: dict[str, int], variables: dict[str, Variable]) -> Command:
    parts = line.split(" ")
    opcode = Opcode.from_string(parts[0])

    if opcode in [Opcode.JMP, Opcode.JZ, Opcode.JNZ, Opcode.CALL]:
        return Command(opcode, labels[parts[1]])
    elif opcode == Opcode.LIT:
        if is_integer(parts[1]):
            value = int(parts[1])
            assert (
                    MIN_NUMBER <= value <= MAX_NUMBER
            ), f"Value {value} is out of bound"
        else:
            value = variables[parts[1]].address
        return Command(opcode, value)
    else:
        return Command(opcode)


def translate_section_text_stage_2(section_text: str, labels: dict[str, int], variables: dict[str, Variable]):
    lines = section_text.splitlines()
    commands: list[Command] = []

    for line in lines:
        commands.append(translate_command(line, labels, variables))

    return commands


def translate_source(source: str) -> tuple[str, str]:
    section_data_index = source.find(SECTION_DATA)
    section_text_index = source.find(SECTION_TEXT)
    section_data = source[section_data_index + len(SECTION_DATA) + 1: section_text_index]
    section_text = source[section_text_index + len(SECTION_TEXT) + 1:]

    variables = {
        "in": Variable("in", INPUT_PORT_ADDRESS, [0], False),
        "out": Variable("out", OUTPUT_PORT_ADDRESS, [0], False)
    }

    variables, section_text_address = translate_section_data(section_data, 4, variables)
    section_text, labels = translate_section_text_stage_1(section_text, section_text_address)
    commands = translate_section_text_stage_2(section_text, labels, variables)

    char_for_index = len(str(section_text_address + 2 * len(
        commands)))  # Нужно для выравнивания индексов машинных слов в коде с комментариями

    code: list[str] = [value_to_binary32(section_text_address)]
    commented_code: list[str] = [f"0{" " * (char_for_index - 1)} {value_to_binary32(section_text_address)}"]

    if INTERRUPTION_1 in labels:
        code.append(value_to_binary32(labels[INTERRUPTION_1]))
        commented_code.append(f"1{" " * (char_for_index - 1)} {value_to_binary32(labels[INTERRUPTION_1])}")
    else:
        code.append(value_to_binary32(0))
        commented_code.append(f"1{" " * (char_for_index - 1)} {value_to_binary32(0)}")

    address = 2

    for variable in variables.values():
        if variable.name in ["in", "out"]:
            code.append(value_to_binary32(0))
            commented_code.append(f"{address}{" " * (char_for_index - len(str(address)))} {value_to_binary32(0)}")
            address += 1
            continue
        for cell in variable.data:
            if variable.is_string and cell != 0:
                code.append(value_to_binary32(cell))
                commented_code.append(
                    f"{address}{" " * (char_for_index - len(str(address)))} {value_to_binary32(cell)} '{chr(cell)}'")
            elif variable.is_string and cell == 0:
                code.append(value_to_binary32(cell))
                commented_code.append(
                    f"{address}{" " * (char_for_index - len(str(address)))} {value_to_binary32(cell)} 0")
            else:
                code.append(value_to_binary32(cell))
                commented_code.append(
                    f"{address}{" " * (char_for_index - len(str(address)))} {value_to_binary32(cell)} {cell}")
            address += 1
    for command in commands:
        code.append(command_to_binary32(command))
        commented_code.append(
            f"{address}{" " * (char_for_index - len(str(address)))} {command_to_binary32(command)} {command.opcode.mnemonic}")
        address += 1
        if command.operand != None:
            code.append(value_to_binary32(command.operand))
            commented_code.append(
                f"{address}{" " * (char_for_index - len(str(address)))} {value_to_binary32(command.operand)} {command.operand}")
            address += 1

    assert (
            address < MEMORY_SIZE
    ), "This programm is too big for processor's memory"

    return "\n".join(code), "\n".join(commented_code)


def main(source, target):
    with open(source, mode="r", encoding="utf-8") as f:
        source = f.read()

    source = clean_source(source)
    code, commented_code = translate_source(source)

    write_commented_code(target[:target.rfind("/") + 1] + "commented_" + target[target.rfind("/") + 1:target.find(".")] + ".txt", commented_code)
    write_code(target, code)

    print(f"source LoC: {len(source.splitlines())} code instr: {len(code.splitlines())}")


if __name__ == "__main__":
    assert (
            len(sys.argv) == 3
    ), "Invalid usage: python3 translator.py <source_file> <target_file>"
    _, source, target = sys.argv
    main(source, target)
