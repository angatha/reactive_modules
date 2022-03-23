import csv
import enum
import itertools
import random
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from typing import TextIO, Callable, NamedTuple, Iterable, Mapping, Iterator, Optional, TypeVar

from tabulate import tabulate

from reactive_modules.cst import Atom, Domain, ReferenceExpression, GuardedCommand, \
    ExecutionContext, EvaluationException, VariableValue, Executable, ScopedVariable, ScopedAtom, VariableResolverChain
from reactive_modules.meta_classes import SingletonFromAbc
from reactive_modules.text_io import read_yes_no, read_number, read_value

T = TypeVar('T')


class ExecutionException(Exception):
    pass


class SimulationStateObserver:
    def on_init_round_start(self, context: ExecutionContext):
        pass

    def on_init_round_ended(self, context: ExecutionContext):
        pass

    def on_round_start(self, context: ExecutionContext):
        pass

    def on_round_end(self, context: ExecutionContext):
        pass

    def on_error(self, ex: EvaluationException | ExecutionException):
        pass

    def before_micro_round(self, chain: VariableResolverChain, atom: Atom, context: ExecutionContext):
        pass

    def on_micro_round_end(self, chain: VariableResolverChain, atom: Atom, chosen_command: GuardedCommand,
                           context: ExecutionContext):
        pass

    def on_end_simulation(self, context: ExecutionContext):
        pass

    def on_simulation_ended(self):
        pass


@dataclass(slots=True)
class ObserverGroup(SimulationStateObserver):
    _observers: tuple[SimulationStateObserver]

    def __init__(self, *observers: SimulationStateObserver):
        self._observers = observers

    def on_init_round_start(self, context: ExecutionContext):
        for observer in self._observers:
            observer.on_init_round_start(context)

    def on_init_round_ended(self, context: ExecutionContext):
        for observer in self._observers:
            observer.on_init_round_ended(context)

    def on_round_start(self, context: ExecutionContext):
        for observer in self._observers:
            observer.on_round_start(context)

    def on_round_end(self, context: ExecutionContext):
        for observer in self._observers:
            observer.on_round_end(context)

    def on_error(self, ex: EvaluationException | ExecutionException):
        for observer in self._observers:
            observer.on_error(ex)

    def before_micro_round(self, chain: VariableResolverChain, atom: Atom, context: ExecutionContext):
        for observer in self._observers:
            observer.before_micro_round(chain, atom, context)

    def on_micro_round_end(self, chain: VariableResolverChain, atom: Atom, chosen_command: GuardedCommand,
                           context: ExecutionContext):
        for observer in self._observers:
            observer.on_micro_round_end(chain, atom, chosen_command, context)


class OutputFormat(str, enum.Enum):
    HTML = 'html'
    PLAIN = 'plain'
    SIMPLE = 'simple'
    GITHUB = 'github'
    MEDIAWIKI = 'mediawiki'
    LATEX = 'latex'

    def __str__(self):
        return self.value


def _print_state(text_out: TextIO, output_format: OutputFormat, context: ExecutionContext, print_private: bool):
    executable = context.executable
    content: list[tuple[str, VariableValue, VariableValue]] = []
    for variable, _ in executable.get_external_variables():
        content.append((variable.full_name,
                        context.get(variable, True),
                        context.get(variable, False)))
    for chain, atom, variable in executable.controlled_variables_with_atoms_in_execution_order():
        scoped_variable = chain.get_renamed(variable)
        if print_private or scoped_variable.variable in chain.head_resolver.observable_variables:
            content.append((scoped_variable.name_including_atom(atom),
                            context.get(scoped_variable, True),
                            context.get(scoped_variable, False)))
    state = tabulate(content,
                     headers=['Variable', 'old', 'new'],
                     colalign=['left', 'right', 'right'],
                     tablefmt=output_format)
    text_out.write(state + '\n')


@dataclass(slots=True)
class IOStateLogger(SimulationStateObserver):
    text_out: TextIO
    output_format: OutputFormat = field(default=OutputFormat.SIMPLE, kw_only=True)
    log_decisions: bool = field(default=True, kw_only=True)
    log_round_status: bool = field(default=True, kw_only=True)
    log_error: bool = field(default=True, kw_only=True)
    log_state_after_round: bool = field(default=True, kw_only=True)
    log_state_before_micro_round: bool = field(default=False, kw_only=True)

    def on_init_round_start(self, context: ExecutionContext):
        if self.log_round_status:
            self.text_out.write('Start initialize round\n')

    def on_init_round_ended(self, context: ExecutionContext):
        if self.log_state_after_round or self.log_round_status:
            self.text_out.write('Initialize round ended:\n')
        if self.log_state_after_round:
            self._print_state(context)

    def on_round_start(self, context: ExecutionContext):
        if self.log_round_status:
            self.text_out.write(f'Start round {context.current_round}\n')

    def on_round_end(self, context: ExecutionContext):
        if self.log_state_after_round or self.log_round_status:
            self.text_out.write(f'Round {context.current_round} ended:\n')
        if self.log_state_after_round:
            self._print_state(context)

    def on_error(self, ex: EvaluationException | ExecutionException):
        if self.log_error:
            self.text_out.write(f'{type(ex).__name__}: {ex}\n')

    def before_micro_round(self, chain: VariableResolverChain, atom: Atom, context: ExecutionContext):
        if self.log_state_before_micro_round:
            self._print_state(context)

    def on_micro_round_end(self, chain: VariableResolverChain, atom: Atom, chosen_command: GuardedCommand,
                           context: ExecutionContext):
        if self.log_decisions:
            self.text_out.write(f'execute for {chain.get_renamed(atom).full_name}: '
                                f'{chosen_command.pretty_print(max_target_length=100000)}\n')

    def _print_state(self, context: ExecutionContext):
        _print_state(self.text_out, self.output_format, context, True)


class LogEntryPart(NamedTuple):
    variable: ScopedVariable
    atom: Atom | None
    value: VariableValue

    @property
    def short_variable_name(self) -> str:
        return self.variable.variable

    @property
    def long_variable_name(self) -> str:
        if self.atom:
            return self.variable.name_including_atom(self.atom)
        return self.variable.full_name


@dataclass(slots=True)
class LogEntry(Mapping[str, str]):
    round: int
    atom: ScopedAtom | None
    values: list[LogEntryPart]

    def __getitem__(self, k: str) -> str:
        if not isinstance(k, str):
            raise TypeError(k)
        if k == 'round':
            return str(self.round)
        elif k == 'atom':
            return self.atom.full_name
        for value in self.values:
            if k == value.long_variable_name:
                return str(value.value)
        raise KeyError(k)

    def __len__(self) -> int:
        return 1 + len(self.values) + (1 if self.atom else 0)

    def __iter__(self) -> Iterator[str]:
        prefix = ('round', 'atom') if self.atom else ('round',)
        return itertools.chain(prefix, (
            value.long_variable_name
            for value in self.values
        ))


@dataclass(slots=True)
class HistoryLogger(SimulationStateObserver):
    log_state_for_micro_round: bool = field(default=False, kw_only=True)
    _history: list[LogEntry] = field(default_factory=list, kw_only=True)

    @property
    def history(self) -> Iterable[LogEntry]:
        yield from self._history

    def on_init_round_ended(self, context: ExecutionContext):
        if not self.log_state_for_micro_round:
            self.append_log_entry(None, context)

    def on_round_end(self, context: ExecutionContext):
        if not self.log_state_for_micro_round:
            self.append_log_entry(None, context)

    def on_micro_round_end(self, chain: VariableResolverChain, atom: Atom, chosen_command: GuardedCommand,
                           context: ExecutionContext):
        if self.log_state_for_micro_round:
            self.append_log_entry(atom, context)

    def on_end_simulation(self, context: ExecutionContext):
        if self.log_state_for_micro_round:
            *_, (_, atom) = context.executable.atoms_in_execution_order()
            self.append_log_entry(atom, context)

    def append_log_entry(self, atom: Atom | None, context: ExecutionContext) -> LogEntry:
        executable = context.executable
        parts: list[LogEntryPart] = []
        for variable, _ in executable.get_external_variables():
            parts.append(LogEntryPart(variable, None, context.get_newest(variable)))
        for chain, atom, variable in executable.controlled_variables_with_atoms_in_execution_order():
            renamed = chain.get_renamed(variable)
            parts.append(LogEntryPart(renamed, atom, context.get_newest(renamed)))

        new_entry = LogEntry(context.current_round, atom, parts)
        self._history.append(new_entry)
        return new_entry

    def get_history(self) -> Iterable[Iterable[str]]:
        columns: dict[str, int] = dict(map(lambda a: (a[1], a[0]), enumerate(sorted(set((
            value.long_variable_name
            for entry in self._history
            for value in entry.values
        ))), start=1)))
        header = itertools.chain(('round',), columns)
        yield header

        for entry in self._history:
            row = list(itertools.repeat('?', 1 + len(columns)))
            row[0] = entry.round
            for value in entry.values:
                row[columns[value.long_variable_name]] = str(value.value)
            yield row


@dataclass(slots=True)
class IOCsvHistoryLogger(HistoryLogger):
    data_out: TextIO = sys.stdout
    _csv_out: csv.DictWriter = field(init=False)

    def on_init_round_start(self, context: ExecutionContext):
        super(IOCsvHistoryLogger, self).on_init_round_start(context)
        columns = sorted(itertools.chain((
            variable.full_name
            for variable, _ in context.executable.get_external_variables()
        ), (
            chain.get_renamed(variable).name_including_atom(atom)
            for chain, atom, variable in
            context.executable.controlled_variables_with_atoms_in_execution_order()
        )))
        fieldnames = ['round', *columns]
        if self.log_state_for_micro_round:
            fieldnames.insert(1, 'atom')
        self._csv_out = csv.DictWriter(self.data_out, fieldnames=fieldnames, restval='?', dialect=csv.unix_dialect)
        self._csv_out.writeheader()

    def append_log_entry(self, atom: Atom | None, context: ExecutionContext) -> LogEntry:
        new_entry = super(IOCsvHistoryLogger, self).append_log_entry(atom, context)
        self._csv_out.writerow(new_entry)
        return new_entry

    def on_simulation_ended(self):
        super(IOCsvHistoryLogger, self).on_simulation_ended()
        self.data_out.close()


def choice(statements: list[GuardedCommand]) -> GuardedCommand:
    """Select one guarded statement from the list. The probability of one being chosen is equal to its options.

    :param statements: available statements
    :return: one statement from the list.
    """
    return random.choices(statements, [s.possible_options for s in statements])[0]


class SimulationController(ABC):
    @abstractmethod
    def choose(self, context: ExecutionContext, chain: VariableResolverChain, atom: Atom,
               options: list[GuardedCommand]) -> Optional[GuardedCommand]:
        pass

    @abstractmethod
    def read_external(self, context: ExecutionContext, variable: ScopedVariable, domain: Domain) -> VariableValue:
        pass

    @abstractmethod
    def wants_change_external_variables(self, context: ExecutionContext):
        pass


class EndlessController(SimulationController, metaclass=SingletonFromAbc):
    def choose(self, context: ExecutionContext, chain: VariableResolverChain, atom: Atom,
               options: list[GuardedCommand]) -> Optional[GuardedCommand]:
        return choice(options)

    def read_external(self, context: ExecutionContext, variable: ScopedVariable, domain: Domain) -> VariableValue:
        return domain.generate_random()

    def wants_change_external_variables(self, context: ExecutionContext):
        return True


@dataclass(slots=True, frozen=True)
class NRoundsController(SimulationController):
    n: int

    def choose(self, context: ExecutionContext, chain: VariableResolverChain, atom: Atom,
               options: list[GuardedCommand]) -> Optional[GuardedCommand]:
        if 0 <= self.n < context.current_round:
            return None
        return choice(options)

    def read_external(self, context: ExecutionContext, variable: ScopedVariable, domain: Domain) -> VariableValue:
        return domain.generate_random()

    def wants_change_external_variables(self, context: ExecutionContext):
        return True


@dataclass(slots=True)
class InteractiveController(SimulationController, SimulationStateObserver):
    command_in: TextIO
    command_out: TextIO
    skip_round = -1
    output_format = OutputFormat.SIMPLE
    end_simulation: bool = False

    def on_init_round_ended(self, context: ExecutionContext):
        self.command_out.write('Init round ended\n')

    def on_round_end(self, context: ExecutionContext):
        self.command_out.write(f'Round {context.current_round} ended\n')

    def choose(self, context: ExecutionContext, chain: VariableResolverChain, atom: Atom,
               options: list[GuardedCommand]) -> Optional[GuardedCommand]:
        if self.end_simulation:
            return None
        if self.skip_round >= context.current_round:
            return choice(options)
        additional_option_str = 'Enter i to print the current state, s to choose random until the next round \n' \
                                'and s followed by a positive number to skip s rounds s1 is equivalent to s.'

        def additional(data: str) -> bool:
            if data == 'i':
                return True
            elif data == 's':
                return True
            elif data.startswith('s'):
                try:
                    return int(data[1:]) > 0
                except ValueError:
                    pass
            return False

        print_private = False

        def ask() -> None | str | GuardedCommand:
            _print_state(self.command_out, self.output_format, context, print_private)
            try:
                extended_options = [g for option in options for g in option.flattened]

                if len(extended_options) == 1:
                    msg = extended_options[0].pretty_print(chain=chain, max_target_length=10000)
                    i = read_yes_no(self.command_in, self.command_out, True,
                                    lambda x: f'Only one option. Continue with atom '
                                              f'{chain.get_renamed(atom).full_name}? ({x}): {msg}\n'
                                              f'{additional_option_str}', additional=additional)
                    if isinstance(i, str):
                        return i
                    elif i:
                        return extended_options[0]
                    else:
                        return None
                else:
                    str_options = [g.pretty_print(chain=chain, max_target_length=10000) for g in extended_options]

                    i = read_number(self.command_in, self.command_out,
                                    f'Got multiple options for atom {chain.get_renamed(atom).full_name}',
                                    f'Which one do you want to use?\n{additional_option_str}', str_options,
                                    allow_none=True, allow_none_str='Press enter for a random one.',
                                    additional=additional)
                    if isinstance(i, str):
                        return i
                    return extended_options[i] if i is not None else choice(options)
            except KeyboardInterrupt:
                return None

        while True:
            match ask():
                case 'i':
                    print_private = True
                    pass
                case 's':
                    self.skip_round = context.current_round
                    return self.choose(context, chain, atom, options)
                case str(n):
                    self.skip_round = context.current_round + int(n[1:]) - 1
                    return self.choose(context, chain, atom, options)
                case res:
                    return res

    def read_external(self, context: ExecutionContext, variable: ScopedVariable, domain: Domain) -> VariableValue:
        if self.end_simulation:
            return domain.generate_random()
        try:
            return read_value(self.command_in, self.command_out, variable.full_name, domain, True)
        except KeyboardInterrupt:
            self.end_simulation = True
            return domain.generate_random()

    def wants_change_external_variables(self, context: ExecutionContext):
        if self.end_simulation:
            return False
        _print_state(self.command_out, self.output_format, context, False)
        try:
            return read_yes_no(self.command_in, self.command_out, False,
                               lambda x: f'Do you want to change the external variables? {x}')
        except KeyboardInterrupt:
            self.end_simulation = True
            return False


def _register_observer_as_error_listener(func: T) -> T:
    @wraps(func)
    def wrapper(*args, observer: SimulationStateObserver, **kwargs):
        try:
            return func(*args, observer=observer, **kwargs)
        except (EvaluationException, ExecutionException) as e:
            observer.on_error(e)
            raise
        finally:
            observer.on_simulation_ended()

    return wrapper


@_register_observer_as_error_listener
def simulate_module(executable: Executable, *, controller: SimulationController = EndlessController(),
                    observer: SimulationStateObserver = SimulationStateObserver()):
    context = ExecutionContext(executable)

    keep_running = True

    def _run_round(command_extractor: Callable[[Atom], Iterable[GuardedCommand]]) -> bool:
        nonlocal keep_running
        if context.in_init or controller.wants_change_external_variables(context):
            for variable, domain in sorted(executable.get_external_variables(), key=lambda v: v[0].full_name):
                context.new_values[variable] = controller.read_external(context, variable, domain)
        for chain, atom in executable.atoms_in_execution_order():
            options: list[GuardedCommand] = [
                command
                for command in command_extractor(atom)
                if command.predicate.eval(chain, context)
            ]
            if not options:
                variables = sorted(set((
                    f'{ref.pretty_print(chain=chain)} => {context[chain, ref]}'
                    for command in command_extractor(atom)
                    for ref in command.predicate.find_nodes(lambda n: isinstance(n, ReferenceExpression))
                )))
                raise ExecutionException(f'No predicate of {"init" if context.in_init else "update"} block of atom '
                                         f'{chain.get_renamed(atom).full_name} matched for variables: '
                                         f'{", ".join(variables)}')
            observer.before_micro_round(chain, atom, context)
            chosen_command = controller.choose(context, chain, atom, options)
            if chosen_command is None:
                keep_running = False
                break
            chosen_command.run(chain, context)
            observer.on_micro_round_end(chain, atom, chosen_command, context)
            context.end_micro_round(chain, atom)
        return keep_running

    observer.on_init_round_start(context)
    if _run_round(lambda a: a.init):
        observer.on_init_round_ended(context)
    while keep_running:
        # ist here to end init
        context.next_round()
        observer.on_round_start(context)
        if _run_round(lambda a: a.effective_update):
            observer.on_round_end(context)
    observer.on_end_simulation(context, )
