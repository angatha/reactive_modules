from dataclasses import dataclass, field
from typing import Optional


@dataclass(order=True, unsafe_hash=True)
class Token(object):
    token: str = field(hash=True)
    line: int = field(compare=False, hash=False)
    start: int = field(compare=False, hash=False)
    end: int = field(compare=False, hash=False)

    def __repr__(self) -> str:
        return f'"{self.token}"({self.line}:{self.start}-{self.end})'


class SavePoint(object):
    __slots__ = [
        'tokenizer',
        'current_line',
        'current_position',
        'current_position_in_line',
        'new_position',
        'last_token',
    ]

    def __init__(self, tokenizer: 'Tokenizer'):
        self.tokenizer = tokenizer
        self.current_line = None
        self.current_position = None
        self.current_position_in_line = None
        self.new_position = None
        self.last_token = None


class Tokenizer(object):
    delimiters = set(' \t\n\f')
    single_character_terminals = set(':;{},→∧\u2227&∨\u2228|><≥≤+-*/()=¬![]')
    multi_character_terminals = [
        '[]',
        '==',
        '!=',
        '/=',
        '>=',
        '<=',
        '->',
        ':=',
        '||',
    ]

    def __init__(self, source_file_content: str, first_line: int = 1):
        source_file_content = source_file_content.replace('\r\n', '\n') \
            .replace('\r', '\n')
        self._content = source_file_content
        self._current_line = first_line
        self._current_position = 0
        self._current_position_in_line = 1
        # used to temporary store the new position in the case has_next_token is called
        self._new_position = -1
        self._max_position = len(source_file_content)
        self.last_token = None

    @property
    def line(self) -> int:
        return self._current_line

    @property
    def position(self) -> int:
        return self._current_position_in_line

    def save(self) -> SavePoint:
        savepoint = SavePoint(self)
        savepoint.current_line = self._current_line
        savepoint.current_position = self._current_position
        savepoint.current_position_in_line = self._current_position_in_line
        savepoint.new_position = self._new_position
        savepoint.last_token = self.last_token
        return savepoint

    def restore(self, savepoint: SavePoint):
        if savepoint.tokenizer is not self:
            raise ValueError('Savepoint is not from this tokenizer')
        self._current_line = savepoint.current_line
        self._current_position = savepoint.current_position
        self._current_position_in_line = savepoint.current_position_in_line
        self._new_position = savepoint.new_position
        self.last_token = savepoint.last_token

    def has_more_token(self) -> bool:
        if 0 <= self._new_position < self._max_position:
            return True
        # Find next token and compare it with the current position
        self._new_position = self.skip_delimiters(self._current_position)
        return self._new_position < self._max_position

    def next_token(self) -> Token:
        # If we already know the start position of the next token, we can reuse it.
        self._current_position = self._new_position \
            if self._new_position >= 0 \
            else self.skip_delimiters(self._current_position)
        self._new_position = -1
        if self._current_position >= self._max_position:
            raise StopIteration
        line = self._current_line
        start = self._current_position
        line_start = self._current_position_in_line
        token = self.try_match_multi_character_terminals(self._current_position)
        if token is None:
            self.scan_token()
        else:
            self._current_position += len(token)
            self._current_position_in_line += len(token)
        end = self._current_position
        line_end = self._current_position_in_line
        self.last_token = Token(self._content[start: end], line, line_start - 1, line_end - 1)
        return self.last_token

    def next_as_single_line_python_expression(self, end_token: str, increment_token: Optional[str] = None) -> Token:
        # do not use _new_position since we do not parse according to the delimiter setting
        self._new_position = -1
        if self._current_position >= self._max_position:
            raise StopIteration
        line = self._current_line
        start = self._current_position
        line_start = self._current_position_in_line
        if increment_token is not None:
            if not self.scan_with_stack(decrement_token=end_token, increment_token=increment_token,
                                        allow_new_line=False):
                raise StopIteration
        elif not self.scan_until(end_token):
            raise StopIteration
        end = self._current_position
        line_end = self._current_position_in_line
        return Token(self._content[start:end], line, line_start - 1, line_end - 1)

    @staticmethod
    def _reindent_python_code(content: str, first_line_offset: int, line: int) -> str:
        minimum_offset = float('inf')
        lines = content.splitlines()
        if not lines:
            return ''
        if lines[0].strip():
            # add whitespace
            lines[0] = first_line_offset * ' ' + lines[0]
        else:
            lines = lines[1:]
        if not lines:
            return ''
        if not lines[-1].strip():
            lines = lines[:-1]
        for line_number, c in enumerate(lines):
            if c.startswith('\t'):
                raise ValueError(f'Python code must use spaces for indentation but line {line + line_number} '
                                 f'starts with tab')
            leading_spaces = len(c) - len(c.lstrip(' '))
            minimum_offset = min(minimum_offset, leading_spaces)
        return '\n'.join(c[minimum_offset:] for line_number, c in enumerate(lines))

    def skip_delimiters(self, start_position: int) -> int:
        """
        Skips delimiters starting from the specified position. Returns the index of
        the first non-delimiter character at or after start_position. current_line
        and current_position_in_line are updated.
        """
        position = start_position
        while position < self._max_position:
            next_char = self._content[position]
            if self._is_comment_start(position):
                position = self._skip_comment(position)
                continue
            if next_char not in self.delimiters:
                break
            position += 1
            if next_char == '\n':
                self._current_line += 1
                self._current_position_in_line = 1
            else:
                self._current_position_in_line += 1
        return position

    def scan_token(self):
        can_be_single_character_terminal = True
        is_single_character_terminal = False
        while self._current_position < self._max_position:
            next_char = self._content[self._current_position]
            if next_char in self.delimiters or self._is_comment_start(self._current_position):
                break
            if next_char in self.single_character_terminals:
                if can_be_single_character_terminal:
                    # postpone break to update the position
                    is_single_character_terminal = True
                else:
                    break
            if not is_single_character_terminal and \
                    self.try_match_multi_character_terminals(self._current_position) is not None:
                break
            can_be_single_character_terminal = False
            self._current_position += 1
            # we are still in the same line since \n is a delimiter
            self._current_position_in_line += 1
            if is_single_character_terminal:
                break

    def try_match_multi_character_terminals(self, position: int) -> Optional[str]:
        for token in self.multi_character_terminals:
            if self._content[position:].startswith(token):
                return token
        return None

    def scan_until(self, end_token: str) -> bool:
        while self._current_position < self._max_position:
            next_char = self._content[self._current_position]
            if next_char == end_token:
                return True
            if next_char == '\n':
                return False
            self._current_position += 1
            self._current_position_in_line += 1
        return False

    def scan_with_stack(self, *, decrement_token: str, increment_token: str, allow_new_line: bool) -> bool:
        count = 1
        while self._current_position < self._max_position:
            next_char = self._content[self._current_position]
            if next_char == decrement_token:
                count -= 1
                if count == 0:
                    return True
            elif next_char == increment_token:
                count += 1
            self._current_position += 1
            if next_char == '\n':
                if allow_new_line:
                    self._current_line += 1
                    self._current_position_in_line = 1
                else:
                    raise StopIteration('Reached end of line before matching end token')
            else:
                self._current_position_in_line += 1
        return False

    def _is_comment_start(self, current_position: int) -> bool:
        if current_position + 1 < self._max_position and self._content[current_position] == '/':
            next_char = self._content[current_position + 1]
            return next_char == '/' or next_char == '*'
        return False

    def _skip_comment(self, current_position: int) -> int:
        if current_position + 1 < self._max_position and self._content[current_position] == '/':
            next_char = self._content[current_position + 1]
            if next_char == '/':
                return self._skip_inline_comment(current_position + 2)
            elif next_char == '*':
                self._current_position_in_line += 2
                return self._skip_multiline_comment(current_position + 2)

    def _skip_inline_comment(self, current_position: int) -> int:
        position = current_position
        while position < self._max_position:
            next_char = self._content[position]
            position += 1
            if next_char == '\n':
                break
        self._current_line += 1
        self._current_position_in_line = 1
        return position

    def _skip_multiline_comment(self, current_position: int) -> int:
        position = current_position
        found_end_chars = 0
        while position < self._max_position:
            next_char = self._content[position]
            if next_char == '*':
                found_end_chars = 1
            elif found_end_chars == 1 and next_char == '/':
                found_end_chars = 2
            else:
                found_end_chars = 0
            position += 1
            if next_char == '\n':
                self._current_line += 1
                self._current_position_in_line = 1
            else:
                self._current_position_in_line += 1
            if found_end_chars == 2:
                break
        return position

    def peek(self):
        save_point = self.save()
        try:
            if self.has_more_token():
                return self.next_token()
            else:
                return None
        finally:
            self.restore(save_point)

    def __repr__(self):
        end_index = min(self._current_position + 50, self._max_position)
        return f'{self.__class__.__name__}{{position={self._current_position}, ' \
               f'line={self._current_line}, line position={self._current_position_in_line}, ' \
               f'next_content="{self._content[self._current_position:end_index]}"}}'
