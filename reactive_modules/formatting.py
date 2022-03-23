from reactive_modules.cst import CombinedModule
from reactive_modules.parser import ConsistencyException, parse, ParseException
from reactive_modules.tokenizer import Tokenizer

UNKNOWN_SOURCE = object()


def parse_file(text: str, source: str = UNKNOWN_SOURCE, force_executable: bool = False) -> CombinedModule | list[str]:
    prefix = '' if source is UNKNOWN_SOURCE else f'{source}: '
    tokenizer = Tokenizer(text)
    try:
        return parse(tokenizer, force_executable)
    except ParseException as e:
        errs = [f'{prefix}{e.message}']
        if e.cause is not None:
            errs.append(f'{prefix}{e.cause}')
        return errs
    except ConsistencyException as e:
        return [f'{prefix}{c}' for c in e.all_causes()]


def format_str(text: str, source: str = UNKNOWN_SOURCE, max_target_length: int = 120) -> str | list[str]:
    match parse_file(text, source):
        case list(data):
            return data
        case module:
            return module.pretty_print(max_target_length=max_target_length) if module else ''
