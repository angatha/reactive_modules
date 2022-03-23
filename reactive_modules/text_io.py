import math
from typing import TextIO, Callable, Optional

from reactive_modules import Domain, VariableValue


def read_yes_no(data_in: TextIO, data_out: TextIO, default_value: bool, message_format: Callable[[str], str],
                additional: Callable[[str], bool] = None) -> bool | str:
    choice = 'Yn' if default_value else 'yN'
    while True:
        data_out.write(message_format(choice) + '\n')
        match data_in.readline():
            case 'y\n' | 'Y\n':
                return True
            case 'n\n' | 'N\n':
                return False
            case '\n':
                return default_value
            case n:
                n = n.strip()
                if additional and additional(n):
                    return n
        data_out.write('Invalid input. Try again.\n')


def read_number(data_in: TextIO, data_out: TextIO, prefix: str, suffix: str, options: list[str], *,
                allow_none: bool = False, allow_none_str: str = 'Press enter for no one.',
                additional: Callable[[str], bool] = None) -> Optional[int | str]:
    digits = math.ceil(math.log10(len(options)))
    msg = prefix
    for i, option in enumerate(options, start=1):
        msg += f'\n  {i: {digits}}. {option}'
    msg += f'\n{suffix}\n'
    if allow_none:
        msg += f'{allow_none_str}\n'
    while True:
        data_out.write(msg)
        line = data_in.readline().strip()
        if allow_none and not line:
            return None
        if additional and additional(line):
            return line
        try:
            n = int(line)
            if 0 < n <= len(options):
                return n - 1
        except ValueError:
            pass
        data_out.write('Invalid input. Try again.\n')


def read_value(data_in: TextIO, data_out: TextIO, target: str, domain: Domain, enter_for_random: bool) -> VariableValue:
    msg = f'Enter the new value for "{target}" chosen from {domain}'
    if enter_for_random:
        msg += '. Press enter for a random one'
    msg += ':\n'
    while True:
        data_out.write(msg)
        stripped_line = data_in.readline().strip()
        if not stripped_line and enter_for_random:
            random = domain.generate_random()
            data_out.write(f'"{random}" was chosen for "{target}.\n')
            return random
        # parse value
        match stripped_line.lower():
            case 'true':
                value = True
            case 'false':
                value = False
            case _:
                try:
                    value = int(stripped_line)
                except ValueError:
                    try:
                        value = float(stripped_line)
                    except ValueError:
                        value = stripped_line
        if domain.contains_value(value):
            return value
        data_out.write(f'"{stripped_line}" is not in {domain}.\n')
