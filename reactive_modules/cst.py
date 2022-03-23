import enum
import graphlib
import itertools
import operator
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from functools import wraps, lru_cache
from typing import Any, Callable, Optional, Iterable, TypeVar, overload, NamedTuple

from reactive_modules.meta_classes import SingletonFromAbc
from reactive_modules.tokenizer import Token

VariableValue = str | int | float | bool


class Qualifier(str, enum.Enum):
    private = 'private'
    interface = 'interface'
    external = 'external'


@dataclass(frozen=True, order=True, slots=True)
class ScopedVariable:
    qualifier: Qualifier = field(hash=False)
    scope: str | None = field(hash=True)
    variable: str = field(hash=True)
    full_name: str = field(init=False)

    def __post_init__(self):
        full_name = f'<{self.qualifier}> {"" if self.scope is None else (self.scope + ".")}{self.variable}'
        object.__setattr__(self, 'full_name', full_name)

    def name_including_atom(self, atom: 'Atom') -> str:
        return f'<{self.qualifier}> {"" if self.scope is None else self.scope}.{atom.name}.{self.variable}'

    def __repr__(self) -> str:
        return self.full_name


@dataclass(frozen=True, order=True, slots=True)
class ScopedAtom:
    scope: str | None = field(hash=True)
    atom: 'Atom' = field(hash=True)
    decisions: tuple[str] = field(hash=True, default_factory=tuple)
    full_name: str = field(init=False)

    def __post_init__(self):
        res = self.scope
        for decision in self.decisions:
            res += '.' + decision
        res += '.' + self.atom.name
        object.__setattr__(self, 'full_name', res)

    def new_scope(self, scope: str, decision: str | None = None) -> 'ScopedAtom':
        if decision is not None:
            decisions = decision, *self.decisions
        else:
            decisions = self.decisions
        return ScopedAtom(scope, self.atom, decisions)

    def __repr__(self) -> str:
        return self.full_name


@dataclass(slots=True)
class ExecutionContext:
    executable: 'Executable'
    current_round: int = 0
    old_values: dict[ScopedVariable, VariableValue] = field(default_factory=dict)
    new_values: dict[ScopedVariable, VariableValue] = field(default_factory=dict)

    @property
    def in_init(self) -> bool:
        return self.current_round == 0

    def next_round(self) -> int:
        self.old_values = self.new_values
        self.new_values = {
            k: v for k, v in self.old_values.items() if k.qualifier == Qualifier.external
        }
        self.current_round += 1
        return self.current_round

    def end_micro_round(self, chain: 'VariableResolverChain', atom: 'Atom'):
        if not self.in_init:
            for variable in atom.controlled_variables:
                key = chain.get_renamed(variable)
                self.new_values.setdefault(key, self.old_values[key])

    def set_new(self, chain: 'VariableResolverChain', target_name: str, new_value: VariableValue):
        self.new_values[chain.get_renamed(target_name)] = new_value

    def __setitem__(self, key: 'tuple[ScopedVariable, bool] |'
                               'tuple[VariableResolverChain, ReferenceExpression] |'
                               'tuple[VariableResolverChain, str, bool]', value: VariableValue):
        try:
            match key:
                case (ScopedVariable() as variable, True):
                    self.old_values[variable] = value
                case (ScopedVariable() as variable, False):
                    self.new_values[variable] = value
                case (VariableResolverChain() as chain, ReferenceExpression(name=name, refers_to_old=True)) | \
                     (VariableResolverChain() as chain, str(name), True):
                    self.old_values[chain.get_renamed(name)] = value
                case (VariableResolverChain() as chain, ReferenceExpression(name=name, refers_to_old=False)) | \
                     (VariableResolverChain() as chain, str(name), False):
                    self.new_values[chain.get_renamed(name)] = value
                case _:
                    if isinstance(key, tuple):
                        raise TypeError(f'{_prettify_values_for_exception(key)} is not applicable to '
                                        f'subscribe to a ExecutionContext')
                    raise TypeError(f'{key} is not applicable to subscribe to a ExecutionContext')
        except KeyError:
            raise KeyError(_prettify_values_for_exception(key)) from None

    @overload
    def get(self, variable: ScopedVariable, refers_to_old: bool) -> VariableValue:
        pass

    @overload
    def get(self, chain: 'VariableResolverChain', reference: 'ReferenceExpression') -> VariableValue:
        pass

    @overload
    def get(self, chain: 'VariableResolverChain', variable: str, refers_to_old: bool) -> VariableValue:
        pass

    def get(self, *items):
        try:
            # noinspection PyTypeChecker
            return self[items]
        except KeyError:
            return '?'

    @overload
    def get_newest(self, chain: 'VariableResolverChain', variable: str):
        pass

    @overload
    def get_newest(self, variable: ScopedVariable):
        pass

    def get_newest(self, *args):
        match args:
            case (VariableResolverChain() as chain, str(variable)):
                variable = chain.get_renamed(variable)
            case (ScopedVariable() as variable, ):
                pass
            case _:
                raise ValueError('Illegal argument types.', args)
        try:
            return self[variable, False]
        except KeyError:
            try:
                return self[variable, True]
            except KeyError:
                return '?'

    def __getitem__(self, item: 'tuple[ScopedVariable, bool] |'
                                'tuple[VariableResolverChain, ReferenceExpression] |'
                                'tuple[VariableResolverChain, str, bool]') -> VariableValue:
        try:
            match item:
                case (ScopedVariable() as variable, True):
                    return self.old_values[variable]
                case (ScopedVariable() as variable, False):
                    return self.new_values[variable]
                case (VariableResolverChain() as chain, ReferenceExpression(name=name, refers_to_old=True)) | \
                     (VariableResolverChain() as chain, str(name), True):
                    return self.old_values[chain.get_renamed(name)]
                case (VariableResolverChain() as chain, ReferenceExpression(name=name, refers_to_old=False)) | \
                     (VariableResolverChain() as chain, str(name), False):
                    return self.new_values[chain.get_renamed(name)]
        except KeyError:
            raise KeyError(_prettify_values_for_exception(item))
        if isinstance(item, tuple):
            raise TypeError(f'{_prettify_values_for_exception(item)} is not applicable to '
                            f'subscribe to a ExecutionContext')
        raise TypeError(f'{item} is not applicable to subscribe to a ExecutionContext')


def _prettify_values_for_exception(item: object) -> object:
    if isinstance(item, tuple):
        return tuple((_prettify_value_for_exception(t) for t in item))
    return item


def _prettify_value_for_exception(value):
    match value:
        case Module(name=name):
            name = f'Module({name})'
        case Atom(name=name, module=Module(name=module_name)):
            name = f'Atom({module_name}.{name})'
        case name:
            pass
    return name


class EvaluationException(Exception):
    pass


T1 = TypeVar('T1')
T2 = TypeVar('T2')


def default_eq(func: Callable[[T1, T2], bool]) -> Callable[[T1, T2], bool]:
    @wraps(func)
    def wrapper(self, o: object) -> bool:
        if self is o:
            return True
        if isinstance(o, BracedExpression):
            return self == o.sub_tree
        if type(self) != type(o):
            return False
        return func(self, o)

    return wrapper


class ConsistencyException(Exception):
    def __init__(self, message: str, cause=None):
        super().__init__(message, cause)
        self.message = message
        if cause is None:
            self.causes = []
        elif isinstance(cause, list):
            self.causes = cause
        else:
            self.causes = [cause]

    def add_causes(self, *causes):
        for cause in causes:
            if isinstance(cause, Exception):
                if cause is self:
                    raise ValueError('Can not add self as its own cause')
                self.causes.append(cause)
            else:
                self.add_causes(*cause)

    def has_causes(self) -> bool:
        return bool(self.causes)

    def all_causes(self) -> Iterable[Exception]:
        if self.has_causes():
            for cause in self.causes:
                if hasattr(cause, 'has_causes') and hasattr(cause, 'all_causes') and cause.has_causes():
                    if cause is self:
                        raise ValueError('self was added as its own cause')
                    yield from cause.all_causes()
                else:
                    yield cause
        else:
            yield self

    def __str__(self):
        return self.message


class IncompatibleModulesException(ConsistencyException):

    def __init__(self, source: 'SourceDescriptor', module0: 'ModuleLike', module1: 'ModuleLike',
                 incompatible_private_module0: list[str], incompatible_private_module1: list[str],
                 incompatible_interface: set[str], cyclic_error: ConsistencyException | None):
        msg = f'Parallel composition between {module0.name} and {module1.name} at {source} is not possible.'

        def add_incompatible_private(module: 'ModuleLike', other_module: 'ModuleLike', variables: list[str]):
            nonlocal msg
            match variables:
                case [variable]:
                    msg += f' Module {module.name} has a private variable {variable} that is a variable of module ' \
                           f'{other_module.name}.'
                case []:
                    pass
                case _:
                    msg += f' Module {module.name} at {module.source} has private variables {", ".join(variables)} ' \
                           f'that are variables of module {other_module.name}.'

        add_incompatible_private(module0, module1, incompatible_private_module0)
        add_incompatible_private(module1, module0, incompatible_private_module1)
        if incompatible_interface:
            msg += f' Both modules declare the following interface variables: {incompatible_interface}.'

        super().__init__(msg, cyclic_error)

    def all_causes(self) -> Iterable[Exception]:
        if self.has_causes():
            for cause in self.causes:
                if hasattr(cause, 'has_causes') and hasattr(cause, 'all_causes') and cause.has_causes():
                    yield from cause.all_causes()
                else:
                    yield cause
        yield self


class TypeException(Exception):
    @staticmethod
    def wrong_literal(literal: 'ConstantExpression', domain: 'Domain') -> 'TypeException':
        return TypeException(f'Illegal literal "{literal.value}" at {literal.source} for domain {domain}')

    @staticmethod
    @overload
    def wrong_type(reference: 'ReferenceExpression', expected_domain: 'Domain',
                   actual_domain: 'Domain') -> 'TypeException':
        pass

    @staticmethod
    @overload
    def wrong_type(expression: 'Expression', expected_domain: 'Domain',
                   actual_domain: 'Domain') -> 'TypeException':
        pass

    @staticmethod
    @overload
    def wrong_type(expression: 'Expression', expected_domain: 'Domain') -> 'TypeException':
        pass

    @staticmethod
    def wrong_type(expression: 'Expression', expected_domain: 'Domain',
                   actual_domain: 'Optional[Domain]' = None) -> 'TypeException':
        if isinstance(expression, ReferenceExpression):
            return TypeException(
                f"Can not use variable {expression.formatted()} at {expression.source} for type of domain "
                f"{expected_domain} because it's actual domain is {actual_domain}.")
        msg = f'Can not use expression "{expression.pretty_print(max_target_length=10000)}" at {expression.source} ' \
              f'for type of domain {expected_domain}'
        if actual_domain is not None:
            msg += f" because it's actual domain is {actual_domain}."
        else:
            msg += '.'
        return TypeException(msg)

    @staticmethod
    def domain_missmatch_on_binary_operator(op_code: str, op_source: 'SourceDescriptor', left_domain: 'Domain',
                                            right_domain: 'Domain') -> 'TypeException':
        return TypeException(f"Can not match domains of {op_code} at {op_source}. Left domain is {left_domain}, right "
                             f"domain is {right_domain}.")

    @staticmethod
    def unsupported_operation(op_code: str, op_source: 'SourceDescriptor', domain: 'Domain') -> 'TypeException':
        return TypeException(f"operation {op_code} at {op_source} is not supported for domain {domain}.")


class LightVariableResolverChainLike(ABC):
    @abstractmethod
    def get_renamed_full_name(self, original_name: str) -> str:
        pass


class LightIdentityVariableResolverChain(LightVariableResolverChainLike, metaclass=SingletonFromAbc):
    def get_renamed_full_name(self, original_name: str) -> str:
        return original_name


class SourceDescriptor:
    @overload
    def __init__(self, *, start: 'Token | SourceDescriptor | SourceCode'):
        pass

    @overload
    def __init__(self, *, start: int, start_column: int):
        pass

    @overload
    def __init__(self, *, start: 'Token | SourceDescriptor | SourceCode',
                 end: 'Token | SourceDescriptor | SourceCode | None'):
        pass

    @overload
    def __init__(self, *, start: int, start_column: int, end: 'Token | SourceDescriptor | SourceCode | None'):
        pass

    @overload
    def __init__(self, *, start: 'Token | SourceDescriptor | SourceCode', end: int, end_column: int):
        pass

    @overload
    def __init__(self, *, start: int, start_column: int, end: int, end_column: int):
        pass

    def __init__(self, *, start: 'int | Token | SourceDescriptor | SourceCode', start_column: Optional[int] = None,
                 end: 'int | Token | SourceDescriptor | SourceCode' = None, end_column: Optional[int] = None):
        if isinstance(start, SourceCode):
            start = start.source
        if isinstance(end, SourceCode):
            end = end.source
        if isinstance(start, SourceDescriptor):
            if start_column is not None:
                raise ValueError('If a token is passed as start, start_column must be omitted')
            self._start_line = start.start_line
            self._start_column = start.start_column
        elif isinstance(start, Token):
            if start_column is not None:
                raise ValueError('If a token is passed as start, start_column must be omitted')
            self._start_line = start.line
            self._start_column = start.start
        elif start_column is None:
            raise ValueError('If start is an int, the column must be specified')
        else:
            self._start_line = start
            self._start_column = start_column
        if end is not None:
            if end_column is None:
                if isinstance(end, SourceDescriptor):
                    self._end_line = end.end_line
                    self._end_column = end.end_column
                elif isinstance(end, Token):
                    self._end_line = end.line
                    self._end_column = end.end
                else:
                    raise ValueError('If end was passed but end_column was not, end must be a token')
            else:
                if isinstance(end, int):
                    self._end_line = end
                    self._end_column = end_column
                else:
                    raise ValueError('If end and end_column where passed, both must be int')
        elif end_column is not None:
            raise ValueError('If end_column is given, end is also required')
        else:
            self._end_line = None
            self._end_column = None

    @property
    def start_line(self) -> int:
        return self._start_line

    @property
    def start_column(self) -> int:
        return self._start_column

    @property
    def end_line(self) -> Optional[int]:
        return self._end_line

    @property
    def end_column(self) -> Optional[int]:
        return self._end_column

    @overload
    def set_end(self, end: 'Token | SourceDescriptor | SourceCode'):
        pass

    @overload
    def set_end(self, end: int, end_column: int):
        pass

    def set_end(self, end: 'int | Token | SourceDescriptor | SourceCode', end_column: Optional[int] = None):
        if isinstance(end, SourceCode):
            end = end.source
        if self._end_line is not None:
            raise ValueError('End was already set')
        if isinstance(end, SourceDescriptor):
            if end_column is not None:
                raise ValueError('If a token is passed as end, end_column must be omitted')
            self._end_line = end.end_line
            self._end_column = end.end_column
        elif isinstance(end, Token):
            if end_column is not None:
                raise ValueError('If a token is passed as end, end_column must be omitted')
            self._end_line = end.line
            self._end_column = end.end
        elif end_column is None:
            raise ValueError('If end is an int, the column must be specified')
        else:
            self._end_line = end
            self._end_column = end_column

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'{self.start_line}:{self.start_column}-{self.end_line if self.end_line is not None else "?"}' \
               f':{self.end_column if self.end_column is not None else "?"}'


@dataclass(slots=True)
class SourceCode:
    source: SourceDescriptor = field(compare=False)


class ParseTreeNode(ABC):

    @staticmethod
    def offset(s: str, first_linde_indentation: int) -> int:
        i = s.rfind('\n')
        return len(s) + first_linde_indentation if i == -1 else len(s) - i - 1

    @abstractmethod
    def pretty_print(self, indentation: str = '', first_line_indentation: int = 0, max_target_length: int = 120,
                     chain: LightVariableResolverChainLike = LightIdentityVariableResolverChain()) -> str:
        pass

    def _components(self) -> list[tuple[str, Any]]:
        return [
            (name, getattr(self, name))
            for name in dir(self)
            if not name.startswith('_') and not callable(getattr(self, name))
        ]

    def __repr__(self):
        indentation = '  '
        res = f'{self.__class__.__name__}:\n'
        for name, component in self._components():
            res += f'{indentation}>{name}:\n'
            if isinstance(component, list):
                for c in component:
                    component_representation = repr(c).replace('\n', f'\n{indentation}{indentation}')
                    res += f'{indentation}{indentation}{component_representation}\n'
            else:
                component_representation = repr(component).replace('\n', f'\n{indentation}{indentation}')
                res += f'{indentation}{indentation}{component_representation}\n'
        return res


@dataclass
class _ExpressionBase(ABC):
    domain: 'Domain' = field(init=False)

    @abstractmethod
    def _infer_types(self, module: 'Module') -> Iterable[TypeException]:
        pass

    def _set_type(self, target_domain: 'Domain', module: 'Module') -> Iterable[TypeException]:
        # noinspection PyUnresolvedReferences
        actual_domain = self.domain
        if actual_domain != target_domain:
            # noinspection PyTypeChecker
            yield TypeException.wrong_type(self, target_domain, actual_domain)

    @abstractmethod
    def eval(self, chain: 'VariableResolverChain', context: ExecutionContext) -> VariableValue:
        pass

    def find_nodes(self, predicate: Callable[[Any], bool]) -> 'Iterable[Expression]':
        if predicate(self):
            yield self


@dataclass
class ValueExpression(SourceCode, _ExpressionBase, ParseTreeNode, ABC):
    @abstractmethod
    def compress(self) -> 'ValueExpression':
        pass


@dataclass
class BooleanExpression(SourceCode, _ExpressionBase, ParseTreeNode, ABC):
    def __post_init__(self):
        if hasattr(super(BooleanExpression, self), '__post_init__'):
            # noinspection PyUnresolvedReferences
            super(BooleanExpression, self).__post_init__()
        self.domain = BooleanDomain()

    @abstractmethod
    def compress(self) -> 'BooleanExpression':
        pass


Expression = BooleanExpression | ValueExpression


class FixedValue:
    pass


@dataclass
class TrueFalseExpression(FixedValue, BooleanExpression):
    value: bool

    def _infer_types(self, module: 'Module') -> Iterable[TypeException]:
        yield from ()

    def compress(self) -> 'BooleanExpression':
        return self

    def pretty_print(self, indentation: str = '', first_line_indentation: int = 0, max_target_length: int = 120,
                     chain: LightVariableResolverChainLike = LightIdentityVariableResolverChain()) -> str:
        return str(self.value)

    @default_eq
    def __eq__(self, o: 'TrueFalseExpression'):
        return self.value == o.value

    def eval(self, chain: 'VariableResolverChain', context: ExecutionContext) -> VariableValue:
        return self.value


@dataclass
class NotExpression(BooleanExpression):
    sub_tree: BooleanExpression

    def compress(self) -> 'BooleanExpression':
        self.sub_tree = self.sub_tree.compress()
        match self.sub_tree:
            case TrueFalseExpression(value=value):
                return TrueFalseExpression(self.source, not value)
            case (BracedExpression(sub_tree=EqualExpression(lhs=lhs, rhs=rhs, op_source=op_source))):
                return NotEqualExpression(op_source, lhs, rhs).compress()
            case BracedExpression(sub_tree=NotEqualExpression(lhs=lhs, rhs=rhs, op_source=op_source)):
                return EqualExpression(op_source, lhs, rhs).compress()
            case _:
                return self

    def _infer_types(self, module: 'Module') -> Iterable[TypeException]:
        yield from self.sub_tree._infer_types(module)

    def _set_type(self, target_domain: 'Domain', module: 'Module') -> Iterable[TypeException]:
        yield from super(NotExpression, self)._set_type(target_domain, module)
        yield from self.sub_tree._set_type(target_domain, module)

    def eval(self, chain: 'VariableResolverChain', context: ExecutionContext) -> VariableValue:
        return not self.sub_tree.eval(chain, context)

    def pretty_print(self, indentation: str = '', first_line_indentation: int = 0, max_target_length: int = 120,
                     chain: LightVariableResolverChainLike = LightIdentityVariableResolverChain()) -> str:
        return '!' + self.sub_tree.pretty_print(indentation, first_line_indentation + 1, max_target_length, chain)


@dataclass(slots=True)
class BinaryBooleanExpression(BooleanExpression):
    source: SourceDescriptor = field(init=False)
    op_source: SourceDescriptor
    lhs: BooleanExpression
    rhs: BooleanExpression
    op_code: str
    op: Callable[[bool, bool], bool]
    op_support_check: 'Callable[[Domain], bool]'

    def __post_init__(self):
        if hasattr(super(BinaryBooleanExpression, self), '__post_init__'):
            super(BinaryBooleanExpression, self).__post_init__()
        if not isinstance(self.lhs, BooleanExpression) and not isinstance(self.lhs, ReferenceExpression):
            raise ValueError(f'lhs should be of type BooleanExpression or a ReferenceExpression (for type boolean) but '
                             f'was {self.lhs}')
        if not isinstance(self.rhs, BooleanExpression) and not isinstance(self.rhs, ReferenceExpression):
            raise ValueError(f'rhs should be of type BooleanExpression or a ReferenceExpression (for type boolean) but '
                             f'was {self.rhs}')
        self.source = SourceDescriptor(start=self.lhs.source, end=self.rhs.source)

    @default_eq
    def __eq__(self, o: 'BinaryBooleanExpression') -> bool:
        return self.op_code == o.op_code and self.lhs == o.lhs and self.rhs == o.rhs

    def _infer_types(self, module: 'Module') -> Iterable[TypeException]:
        yield from _infer_types_binary(self, module)

    def pretty_print(self, indentation: str = '', first_line_indentation: int = 0, max_target_length: int = 120,
                     chain: LightVariableResolverChainLike = LightIdentityVariableResolverChain()) -> str:
        res = self.lhs.pretty_print(indentation, first_line_indentation, max_target_length, chain)
        first_line_indentation = ParseTreeNode.offset(res, first_line_indentation) + 2 + len(self.op_code)
        if first_line_indentation + len(self.op_code) + 1 > max_target_length:
            res += '\n' + indentation
            first_line_indentation = len(indentation)
        else:
            res += ' '
            first_line_indentation += 1
        res += f'{self.op_code} '
        first_line_indentation += 1 + len(self.op_code)
        res += self.rhs.pretty_print(indentation, first_line_indentation, max_target_length, chain)
        return res

    @abstractmethod
    def _compress(self, const: TrueFalseExpression, other: BooleanExpression) -> BooleanExpression:
        pass

    @abstractmethod
    def _compress_equal(self, hs: BooleanExpression):
        pass

    # noinspection PyTypeChecker
    def compress(self) -> 'BooleanExpression':
        self.lhs = self.lhs.compress()
        self.rhs = self.rhs.compress()
        if isinstance(self.lhs, TrueFalseExpression):
            if isinstance(self.rhs, TrueFalseExpression):
                return TrueFalseExpression(self.source, self.op(self.lhs.value, self.rhs.value))
            else:
                return self._compress(self.lhs, self.rhs)
        else:
            if isinstance(self.rhs, TrueFalseExpression):
                return self._compress(self.rhs, self.lhs)
            elif self.lhs == self.rhs:
                return self._compress_equal(self.lhs)
            else:
                return self

    def eval(self, chain: 'VariableResolverChain', context: ExecutionContext) -> VariableValue:
        return _eval_binary(self, chain, context)

    def find_nodes(self, predicate: Callable[[Any], bool]) -> 'Iterable[Expression]':
        yield from _find_nodes_binary(self, predicate)


class AndExpression(BinaryBooleanExpression):

    def __init__(self, op_source: SourceDescriptor, lhs: BooleanExpression, rhs: BooleanExpression):
        super().__init__(op_source, lhs, rhs, '&', operator.and_, lambda d: d.allows_and)
        self.__post_init__()

    def _compress(self, const: TrueFalseExpression, other: BooleanExpression) -> BooleanExpression:
        return TrueFalseExpression(self.source, False) if not const.value else other

    def _compress_equal(self, hs: BooleanExpression):
        return hs


class OrExpression(BinaryBooleanExpression):

    def __init__(self, op_source: SourceDescriptor, lhs: BooleanExpression, rhs: BooleanExpression):
        super().__init__(op_source, lhs, rhs, '|', operator.or_, lambda d: d.allows_or)
        self.__post_init__()

    def _compress(self, const: TrueFalseExpression, other: BooleanExpression) -> BooleanExpression:
        return TrueFalseExpression(self.source, True) if const.value else other

    def _compress_equal(self, hs: BooleanExpression):
        return hs


@dataclass(slots=True)
class ConstantExpression(FixedValue, ValueExpression):
    value: str | int | float

    def __post_init__(self):
        if hasattr(super(ConstantExpression, self), '__post_init__'):
            # noinspection PyUnresolvedReferences
            super(ConstantExpression, self).__post_init__()
        # noinspection PyTypeChecker
        self.domain = None

    @default_eq
    def __eq__(self, o: 'ConstantExpression') -> bool:
        return self.value == o.value

    def _infer_types(self, module: 'Module') -> Iterable[TypeException]:
        yield from ()

    def _set_type(self, target_domain: 'Domain', module: 'Module') -> Iterable[TypeException]:
        if not target_domain.contains_value(self.value):
            yield TypeException.wrong_type(self, target_domain)
        self.domain = target_domain
        # noinspection PyUnresolvedReferences
        yield from super(ConstantExpression, self)._set_type(target_domain, module)

    def compress(self) -> 'ValueExpression':
        return self

    def pretty_print(self, indentation: str = '', first_line_indentation: int = 0, max_target_length: int = 120,
                     chain: LightVariableResolverChainLike = LightIdentityVariableResolverChain()) -> str:
        return str(self.value)

    def eval(self, chain: 'VariableResolverChain', context: ExecutionContext) -> VariableValue:
        return self.value


class NumericExpression:
    pass


class ConstantNumericExpression(NumericExpression, ConstantExpression):
    pass


def wrap_constant(source: SourceDescriptor, value: str | int | float) -> ConstantExpression:
    if isinstance(value, str):
        return ConstantExpression(source, value)
    else:
        return ConstantNumericExpression(source, value)


@dataclass(slots=True)
class ReferenceExpression(ValueExpression):
    name: str
    refers_to_old: bool

    def __post_init__(self):
        if hasattr(super(ReferenceExpression, self), '__post_init__'):
            # noinspection PyUnresolvedReferences
            super(ReferenceExpression, self).__post_init__()
        # noinspection PyTypeChecker
        self.domain = None

    def formatted(self, chain: LightVariableResolverChainLike = LightIdentityVariableResolverChain()) -> str:
        name = chain.get_renamed_full_name(self.name)
        return name if self.refers_to_old else f"{name}´"

    @default_eq
    def __eq__(self, o: 'ReferenceExpression') -> bool:
        return self.name == o.name and self.refers_to_old == o.refers_to_old

    def _infer_types(self, module: 'Module') -> Iterable[TypeException]:
        target_domain = module.module_variables[self.name].domain
        yield from self._set_type(target_domain, module)

    def _set_type(self, target_domain: 'Domain', module: 'Module') -> Iterable[TypeException]:
        if self.domain is not None:
            # noinspection PyProtectedMember
            yield from super(ReferenceExpression, self)._set_type(target_domain, module)
        else:
            self.domain = target_domain

    def compress(self) -> 'ValueExpression':
        return self

    def pretty_print(self, indentation: str = '', first_line_indentation: int = 0, max_target_length: int = 120,
                     chain: LightVariableResolverChainLike = LightIdentityVariableResolverChain()) -> str:
        return self.formatted(chain)

    def eval(self, chain: 'VariableResolverChain', context: ExecutionContext) -> VariableValue:
        return context[chain, self]


# noinspection PyProtectedMember
def _infer_types_binary(self, module: 'Module') -> Iterable[TypeException]:
    yield from self.lhs._infer_types(module)
    yield from self.rhs._infer_types(module)
    if self.lhs.domain is None:
        if self.rhs.domain is None:
            raise ValueError(f'Did not expect ot reach this point since compression should evaluate binary operations '
                             f'on two constants. Operation is {self.op_code} at {self.op_source} on {self.lhs.source} '
                             f'and {self.rhs.source}')
        yield from self.lhs._set_type(self.rhs.domain, module)
    elif self.rhs.domain is None:
        yield from self.rhs._set_type(self.lhs.domain, module)
    if self.lhs.domain != self.rhs.domain:
        yield TypeException.domain_missmatch_on_binary_operator(self.op_code, self.op_source, self.lhs.domain,
                                                                self.rhs.domain)
    if not hasattr(self, 'domain') or self.domain is None:
        self.domain = self.lhs.domain
    if not self.op_support_check(self.domain):
        yield TypeException.unsupported_operation(self.op_code, self.op_source, self.domain)


def _eval_binary(self, chain: 'VariableResolverChain', context: ExecutionContext) -> VariableValue:
    lhs = self.lhs.eval(chain, context)
    rhs = self.rhs.eval(chain, context)
    try:
        res = self.op(lhs, rhs)
    except Exception:
        raise EvaluationException(f'Failed to evaluate {lhs} {self.op_code} {rhs} at {self.op_source}')
    if not self.domain.contains_value(res):
        raise EvaluationException(f'Evaluation {self.op_code} at {self.op_source} resulted in an invalid value since '
                                  f'{res} is not in {self.domain}')
    return res


def _find_nodes_binary(self, predicate: Callable[[Any], bool]) -> 'Iterable[Expression]':
    if predicate(self):
        yield self
    yield from self.lhs.find_nodes(predicate)
    yield from self.rhs.find_nodes(predicate)


@dataclass
class BinaryValueExpressionBase(ABC):
    source: SourceDescriptor = field(init=False)
    op_source: SourceDescriptor
    lhs: Expression
    rhs: Expression
    op_code: str
    op: Callable[[Any, Any], bool | Any]
    op_support_check: 'Callable[[Domain], bool]'

    def __post_init__(self):
        if hasattr(super(BinaryValueExpressionBase, self), '__post_init__'):
            # noinspection PyUnresolvedReferences
            super(BinaryValueExpressionBase, self).__post_init__()
        if not isinstance(self.lhs, Expression):
            raise ValueError(f'lhs should be of type Expression but was {self.lhs}')
        if not isinstance(self.rhs, Expression):
            raise ValueError(f'rhs should be of type Expression but was {self.rhs}')
        self.source = SourceDescriptor(start=self.lhs.source, end=self.rhs.source)

    @abstractmethod
    def _wrap_constant(self, source: SourceDescriptor, value: bool | Any):
        pass

    @default_eq
    def __eq__(self, o: 'BinaryValueExpressionBase') -> bool:
        return self.op_code == o.op_code and self.lhs == o.lhs and self.rhs == o.rhs

    def _infer_types(self, module: 'Module') -> Iterable[TypeException]:
        # the mixin does not but all concrete implementations do
        # noinspection PyTypeChecker
        yield from _infer_types_binary(self, module)

    def compress(self):
        self.lhs = self.lhs.compress()
        self.rhs = self.rhs.compress()
        if isinstance(self.lhs, FixedValue) and isinstance(self.rhs, FixedValue):
            # noinspection PyUnresolvedReferences
            return self._wrap_constant(SourceDescriptor(start=self.lhs.source, end=self.rhs.source),
                                       self.op(self.lhs.value, self.rhs.value))
        return self

    def pretty_print(self, indentation: str = '', first_line_indentation: int = 0, max_target_length: int = 120,
                     chain: LightVariableResolverChainLike = LightIdentityVariableResolverChain()) -> str:
        first = self.lhs.pretty_print(indentation, first_line_indentation, max_target_length, chain)
        first_line_indentation = ParseTreeNode.offset(first, first_line_indentation) + 2 + len(self.op_code)
        second = self.rhs.pretty_print(indentation, first_line_indentation, max_target_length, chain)
        return f'{first} {self.op_code} {second}'

    def eval(self, chain: 'VariableResolverChain', context: ExecutionContext) -> VariableValue:
        # noinspection PyTypeChecker
        return _eval_binary(self, chain, context)

    def find_nodes(self, predicate: Callable[[Any], bool]) -> 'Iterable[Expression]':
        yield from _find_nodes_binary(self, predicate)


class BinaryValueExpressionToBooleanExpression(BinaryValueExpressionBase, BooleanExpression):
    def _wrap_constant(self, source: SourceDescriptor, value: bool | Any):
        return TrueFalseExpression(source, value)


class BinaryValueExpression(BinaryValueExpressionBase, ValueExpression):
    def _wrap_constant(self, source: SourceDescriptor, value: bool | Any):
        return ConstantExpression(source, value)


class EqualExpression(BinaryValueExpressionToBooleanExpression):
    def __init__(self, op_source: SourceDescriptor, lhs: ValueExpression, rhs: ValueExpression):
        super().__init__(op_source, lhs, rhs, '==', lambda a, b: a == b, lambda d: True)
        self.__post_init__()


class NotEqualExpression(BinaryValueExpressionToBooleanExpression):
    def __init__(self, op_source: SourceDescriptor, lhs: ValueExpression, rhs: ValueExpression):
        super().__init__(op_source, lhs, rhs, '!=', lambda a, b: a != b, lambda d: True)
        self.__post_init__()


class LessThanExpression(BinaryValueExpressionToBooleanExpression):
    def __init__(self, op_source: SourceDescriptor, lhs: ValueExpression, rhs: ValueExpression):
        super().__init__(op_source, lhs, rhs, '<', operator.lt, lambda d: d.allows_ordering)
        self.__post_init__()


class LessThanOrEqualExpression(BinaryValueExpressionToBooleanExpression):
    def __init__(self, op_source: SourceDescriptor, lhs: ValueExpression, rhs: ValueExpression):
        super().__init__(op_source, lhs, rhs, '<=', operator.le, lambda d: d.allows_ordering)
        self.__post_init__()


class GreaterThanExpression(BinaryValueExpressionToBooleanExpression):
    def __init__(self, op_source: SourceDescriptor, lhs: ValueExpression, rhs: ValueExpression):
        super().__init__(op_source, lhs, rhs, '>', operator.gt, lambda d: d.allows_ordering)
        self.__post_init__()


class GreaterThanOrEqualExpression(BinaryValueExpressionToBooleanExpression):
    def __init__(self, op_source: SourceDescriptor, lhs: ValueExpression, rhs: ValueExpression):
        super().__init__(op_source, lhs, rhs, '>=', operator.ge, lambda d: d.allows_ordering)
        self.__post_init__()


class AddExpression(NumericExpression, BinaryValueExpression):
    def __init__(self, op_source: SourceDescriptor, lhs: ValueExpression, rhs: ValueExpression):
        super().__init__(op_source, lhs, rhs, '+', operator.add, lambda d: d.allows_addition)
        self.__post_init__()


class SubtractExpression(NumericExpression, BinaryValueExpression):
    def __init__(self, op_source: SourceDescriptor, lhs: ValueExpression, rhs: ValueExpression):
        super().__init__(op_source, lhs, rhs, '-', operator.sub, lambda d: d.allows_subtraction)
        self.__post_init__()


class MultiplyExpression(NumericExpression, BinaryValueExpression):
    def __init__(self, op_source: SourceDescriptor, lhs: ValueExpression, rhs: ValueExpression):
        super().__init__(op_source, lhs, rhs, '*', operator.mul, lambda d: d.allows_multiplication)
        self.__post_init__()


class DivideExpression(NumericExpression, BinaryValueExpression):
    def __init__(self, op_source: SourceDescriptor, lhs: ValueExpression, rhs: ValueExpression):
        super().__init__(op_source, lhs, rhs, '/', DivideExpression.div, lambda d: d.allows_division)
        self.__post_init__()

    @staticmethod
    def div(a, b):
        if isinstance(a, int):
            return a // b
        else:
            return a / b


@dataclass(slots=True)
class BracedExpression(SourceCode):
    sub_tree: BooleanExpression | ValueExpression

    def __eq__(self, o: object) -> bool:
        return self.sub_tree == o

    def _infer_types(self, module: 'Module') -> Iterable[TypeException]:
        # noinspection PyProtectedMember
        yield from self.sub_tree._infer_types(module)

    def _set_type(self, target_domain: 'Domain', module: 'Module') -> Iterable[TypeException]:
        # noinspection PyProtectedMember
        yield from self.sub_tree._set_type(target_domain, module)

    def __getattr__(self, item):
        return getattr(self.sub_tree, item)

    # noinspection PyUnusedLocal
    def pretty_print(self, indentation: str = '', first_line_indentation: int = 0, max_target_length: int = 120,
                     chain: LightVariableResolverChainLike = LightIdentityVariableResolverChain()) -> str:
        super_call = self.sub_tree.pretty_print(" " * first_line_indentation, first_line_indentation + 1,
                                                max_target_length, chain)
        return f'({super_call})'

    def compress(self):
        self.sub_tree = self.sub_tree.compress()
        match self.sub_tree:
            case TrueFalseExpression() | ReferenceExpression() | ConstantExpression() | FixedValue():
                return self.sub_tree
            case _:
                return self

    def eval(self, chain: 'VariableResolverChain', context: ExecutionContext) -> VariableValue:
        return self.sub_tree.eval(chain, context)

    def find_nodes(self, predicate: Callable[[Any], bool]) -> 'Iterable[Expression]':
        if predicate(self):
            yield self
        yield from self.sub_tree.find_nodes(predicate)


class BracedBooleanExpression(BracedExpression, BooleanExpression):
    pass


class BracedValueExpression(BracedExpression, ValueExpression):
    pass


@dataclass(slots=True)
class Assignment(SourceCode, ParseTreeNode):
    target_name: str
    expression: 'Expression | tuple[Source, FiniteDomain]'

    # noinspection PyProtectedMember
    def _infer_types(self, module: 'Module') -> Iterable[TypeException]:
        yield from self.expression._infer_types(module)
        target_domain = module.module_variables[self.target_name].domain
        # Maybe it is the assignment of a domain, but it was parsed as a ConstantExpression
        if isinstance(self.expression, ConstantExpression) and not target_domain.contains_value(self.expression.value):
            possible_domain = KnownDomain.name_to_domain.get(self.expression.value)
            if possible_domain is not None:
                if isinstance(possible_domain, FiniteDomain):
                    self.expression = self.expression.source, possible_domain
                    return
                else:
                    raise ConsistencyException(f'Can not use domain {possible_domain.symbol} for broadcasting since it '
                                               f'is not finite.')
        yield from self.expression._set_type(target_domain, module)

    def pretty_print(self, indentation: str = '', first_line_indentation: int = 0, max_target_length: int = 120,
                     chain: LightVariableResolverChainLike = LightIdentityVariableResolverChain()) -> str:
        res = chain.get_renamed_full_name(self.target_name) + "´ := "
        first_line_indentation += len(res)
        match self.expression:
            case (_, KnownDomain(symbol=symbol)):
                res += symbol
            case (_, domain) if hasattr(domain, 'pretty_print'):
                res += domain.pretty_print(' ' * first_line_indentation, first_line_indentation, max_target_length,
                                           chain)
            case (_, domain):
                res += domain.values
            case _:
                res += self.expression.pretty_print(' ' * first_line_indentation, first_line_indentation,
                                                    max_target_length, chain)
        return res

    def run(self, chain: 'VariableResolverChain', context: ExecutionContext):
        if isinstance(self.expression, tuple):
            new_value = random.choice(list(self.expression[1].values))
        else:
            new_value = self.expression.eval(chain, context)
        context.set_new(chain, self.target_name, new_value)

    @property
    def is_identity(self) -> bool:
        return isinstance(self.expression, ReferenceExpression) and self.expression.name == self.target_name

    @property
    def possible_options(self) -> int:
        if isinstance(self.expression, tuple):
            return self.expression[1].cardinality
        else:
            return 1

    def flatten(self) -> 'list[Assignment]':
        if isinstance(self.expression, tuple):
            return [
                Assignment(self.source, self.target_name, ConstantExpression(self.expression[0], v))
                for v in self.expression[1].values
            ]
        else:
            return [self]


@dataclass(slots=True)
class GuardedCommand(SourceCode, ParseTreeNode):
    predicate: BooleanExpression | ReferenceExpression
    assignments: list[Assignment]
    _flattened_assignments: 'list[GuardedCommand] | None' = field(init=False)

    @property
    def flattened(self) -> 'list[GuardedCommand]':
        if hasattr(self, '_flattened_assignments'):
            return self._flattened_assignments
        self._flattened_assignments = [
            GuardedCommand(self.source, self.predicate, a)
            for a in itertools.product(*(a.flatten() for a in self.assignments))
        ]
        return self._flattened_assignments

    # noinspection PyProtectedMember
    def _infer_types(self, module: 'Module') -> Iterable[TypeException]:
        yield from self.predicate._infer_types(module)
        actual_domain = self.predicate.domain
        if actual_domain != BooleanDomain():
            yield TypeException.wrong_type(self.predicate, BooleanDomain(), actual_domain)
        for assignment in self.assignments:
            yield from assignment._infer_types(module)

    def pretty_print(self, indentation: str = '', first_line_indentation: int = 0, max_target_length: int = 120,
                     chain: LightVariableResolverChainLike = LightIdentityVariableResolverChain(),
                     *, arrow_indentation: Optional[int] = None) -> str:
        if arrow_indentation is not None and arrow_indentation < len(indentation):
            raise ValueError(f'{arrow_indentation=} must be at least {len(indentation)=}')
        res = '[] ' + self.predicate.pretty_print(indentation + '   ', first_line_indentation + 3, max_target_length,
                                                  chain)
        res += ' '
        still_in_first_line = '\n' not in res
        current_indentation = ParseTreeNode.offset(res, first_line_indentation)

        if arrow_indentation is not None:
            if arrow_indentation < current_indentation:
                res += '\n'
                current_indentation = 0
            if arrow_indentation > current_indentation:
                missing = arrow_indentation - current_indentation - (
                    first_line_indentation if still_in_first_line else 0)
                res += ' ' * missing
                current_indentation += missing

        current_indentation += 2
        assignment_indentation = ' ' * (current_indentation + 1)
        res += '->'

        for i, assignment in enumerate(self.assignments):
            not_last = i < len(self.assignments) - 1
            can_be_on_new_line = i > 0

            assignment_str = assignment.pretty_print(assignment_indentation, current_indentation + 1,  # for the space
                                                     max_target_length, chain)
            if not_last:
                assignment_str += ';'

            if can_be_on_new_line \
                    and ('\n' in assignment_str or len(assignment_str) + current_indentation > max_target_length):
                res += '\n' + assignment_indentation
                assignment_str = assignment.pretty_print(assignment_indentation, len(assignment_indentation),
                                                         max_target_length, chain)
                if not_last:
                    assignment_str += ';'
            else:
                res += ' '
            res += assignment_str
            current_indentation = ParseTreeNode.offset(res, first_line_indentation)
        return res

    def run(self, chain: 'VariableResolverChain', context: ExecutionContext):
        for assignment in self.assignments:
            assignment.run(chain, context)

    @property
    def is_identity(self) -> bool:
        return all((a.is_identity for a in self.assignments))

    @property
    def possible_options(self) -> int:
        return len(self.flattened)


@dataclass(order=True, unsafe_hash=True, slots=True)
class Atom(SourceCode, ParseTreeNode):
    name: str = field(hash=True)
    module: 'Module' = field(hash=True)
    lazy: bool = field(hash=False, compare=False)
    controlled_variables: list[str] = field(hash=False, compare=False)
    read_variables: list[str] = field(hash=False, compare=False)
    awaited_variables: list[str] = field(hash=False, compare=False)
    init: list[GuardedCommand] = field(hash=False, compare=False)
    update: list[GuardedCommand] = field(hash=False, compare=False)

    def __repr__(self):
        return f'{self.module.name}.{self.name}'

    def _assert_variable_set(self, set_name: str, variables: list[str]):
        unique_variables = set()
        for v in variables:
            if v in unique_variables:
                raise ValueError(f'Got duplicated variable "{v}" in {set_name} of atom {self.name}.')
            unique_variables.add(v)

    def __post_init__(self):
        if hasattr(super(Atom, self), '__post_init__'):
            # noinspection PyUnresolvedReferences
            super(Atom, self).__post_init__()
        self._assert_variable_set('controls', self.controlled_variables)
        self.controlled_variables = sorted(self.controlled_variables)
        self._assert_variable_set('reads', self.read_variables)
        self.read_variables = sorted(self.read_variables)
        self._assert_variable_set('await', self.awaited_variables)
        self.awaited_variables = sorted(self.awaited_variables)

    @property
    def effective_update(self) -> Iterable[GuardedCommand]:
        yield from self.update
        if self.lazy:
            yield GuardedCommand(self.source, TrueFalseExpression(self.source, True), [])

    @property
    def all_commands(self) -> Iterable[GuardedCommand]:
        yield from self.init
        if self.init is not self.update:
            yield from self.update

    @property
    def is_combinatorial(self) -> bool:
        return not self.is_sequential

    @property
    def is_sequential(self) -> bool:
        return len(self.read_variables) > 0

    @property
    def has_conditional_sleep_assignment(self) -> bool:
        return any((c.is_identity for c in self.update))

    @property
    def is_passive(self) -> bool:
        # TODO this is still not sufficient enough
        def predicate_matches(predicate: BooleanExpression | ReferenceExpression) -> bool:
            match predicate:
                case (OrExpression(lhs=lhs, rhs=rhs) | AndExpression(lhs=lhs, rhs=rhs)):
                    return predicate_matches(lhs) and predicate_matches(rhs)
                case (EqualExpression(lhs=ReferenceExpression(name=name0, refers_to_old=True),
                                      rhs=ReferenceExpression(name=name1, refers_to_old=False)) |
                      EqualExpression(lhs=ReferenceExpression(name=name0, refers_to_old=False),
                                      rhs=ReferenceExpression(name=name1, refers_to_old=True))):
                    return name0 == name1
                case _:
                    return False

        return self.is_combinatorial or any((
            c.is_identity and predicate_matches(c.predicate)
            for c in self.update))

    @property
    def is_active(self) -> bool:
        return not self.is_passive

    @property
    def is_lazy(self) -> bool:
        return self.lazy \
               or any((c.is_identity and isinstance(c.predicate, ConstantExpression) and c.predicate.value is True
                       for c in self.update))

    @staticmethod
    def _can_append(prefix: int, label: str, variables: list[str], max_target_length: int) -> bool:
        #            space                commas + spaces
        expected_length = prefix + 1 + len(label) + (2 * len(variables) - 1) + sum(map(len, variables))
        return expected_length <= max_target_length

    def pretty_print(self, indentation: str = '', first_line_indentation: int = 0, max_target_length: int = 120,
                     chain: LightVariableResolverChainLike = LightIdentityVariableResolverChain()) -> str:
        res = ''
        # can we write atom <name> not on the first line?
        current_indentation = first_line_indentation
        if first_line_indentation != len(indentation) \
                and first_line_indentation + 5 + (5 if self.lazy else 0) + len(self.name) > max_target_length:
            res += '\n' + indentation
            current_indentation = len(indentation)
        if self.lazy:
            res += 'lazy '
        res += 'atom ' + self.name
        current_indentation += 5 + len(self.name)

        def append_header_part(label: str, variables: list[str]):
            variables = [chain.get_renamed_full_name(v) for v in variables]
            if not variables:
                return
            nonlocal res, current_indentation
            if not Atom._can_append(current_indentation, label, variables, max_target_length):
                res += '\n' + indentation
                current_indentation = len(indentation)
            else:
                res += ' '
                current_indentation += 1

            res += label
            current_indentation += len(label)
            # can the first identifier be placed after the label?
            if current_indentation + 1 + len(variables[0]) + (1 if len(variables) > 1 else 0) <= max_target_length:
                tokens_indentation = ' ' * (current_indentation + 1)
                res += ' ' + variables[0]
                current_indentation += 1 + len(variables[0])
            else:
                tokens_indentation = indentation
                res += '\n' + indentation + variables[0]
                current_indentation = len(tokens_indentation) + len(variables[0])
            if len(variables) > 1:
                res += ','
                current_indentation += 1

            # for all other tokens except the first one:
            for i, variable in enumerate(variables[1:]):
                has_next = i < len(variables) - 2
                if current_indentation + 1 + len(variables[0]) + (1 if has_next else 0) > max_target_length:
                    # new line required
                    res += '\n' + tokens_indentation
                    current_indentation = len(tokens_indentation)
                else:
                    res += ' '
                    current_indentation += 1
                res += variable
                current_indentation += len(variable)
                if has_next:
                    res += ','
                    current_indentation += 1

        append_header_part('controls', self.controlled_variables)
        append_header_part('reads', self.read_variables)
        append_header_part('awaits', self.awaited_variables)

        def print_guarded_commands(guarded_commands: list[GuardedCommand]):
            nonlocal res, current_indentation
            arrow_indent_indexes: list[int] = []
            indents = []
            current_index = -1
            current_list = None
            for i, guarded_command in enumerate(guarded_commands):
                # add arrow indent for each guarded expression

                part = indentation + '  ' + guarded_command.pretty_print(indentation + '  ', len(indentation) + 2,
                                                                         max_target_length, chain)
                part_before_arrow = part[:part.find('->')]
                arrow_offset = ParseTreeNode.offset(part_before_arrow, first_line_indentation)
                can_connect_with_previous = arrow_offset == len(part_before_arrow)
                if i == 0 or not can_connect_with_previous:
                    current_index += 1
                    current_list = []
                    indents.append(current_list)
                current_list.append(arrow_offset)
                arrow_indent_indexes.append(current_index)

            indents = [max(i) for i in indents]

            for i, guarded_command in enumerate(guarded_commands):
                arrow_indentation = indents[arrow_indent_indexes[i]]
                res += '\n' + indentation + '  ' + \
                       guarded_command.pretty_print(indentation + '  ', len(indentation) + 2, max_target_length, chain,
                                                    arrow_indentation=arrow_indentation)

        res += '\n' + indentation
        current_indentation = len(indentation)
        if self.init is self.update:
            res += 'initupdate'
            print_guarded_commands(self.init)
        else:
            res += 'init'
            print_guarded_commands(self.init)
            res += '\n' + indentation
            current_indentation = len(indentation)
            res += 'update'
            print_guarded_commands(self.update)
        # current_indentation is not correct at this point.

        return res


class Domain(ParseTreeNode, ABC):
    @property
    @abstractmethod
    def allows_subtraction(self) -> bool:
        pass

    @property
    def allows_addition(self) -> bool:
        return self.allows_subtraction

    @property
    def allows_multiplication(self) -> bool:
        return self.allows_division

    @property
    @abstractmethod
    def allows_division(self) -> bool:
        pass

    @property
    @abstractmethod
    def allows_ordering(self) -> bool:
        pass

    @property
    @abstractmethod
    def allows_and(self) -> bool:
        pass

    @property
    @abstractmethod
    def allows_or(self) -> bool:
        return self.allows_or

    @abstractmethod
    def contains_value(self, value: VariableValue) -> bool:
        pass

    @abstractmethod
    def generate_random(self) -> VariableValue:
        pass

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __hash__(self):
        pass


class FiniteDomain(ABC):
    @property
    @abstractmethod
    def values(self) -> set[str | int | float | bool]:
        pass

    @property
    def cardinality(self) -> int:
        return len(self.values)


class DefinedDomain(FiniteDomain, Domain):
    def __init__(self, values: list[str | int | float]):
        self._values = set(values)

    def __repr__(self) -> str:
        return '{' + ', '.join((str(v) for v in self.values)) + '}'

    @property
    def values(self) -> set[str | int | float | bool]:
        return self._values

    @property
    def allows_addition(self) -> bool:
        return True

    @property
    def allows_subtraction(self) -> bool:
        return False

    @property
    def allows_division(self) -> bool:
        return False

    @property
    def allows_ordering(self) -> bool:
        return True

    @property
    def allows_and(self) -> bool:
        return False

    @property
    def allows_or(self) -> bool:
        return False

    def contains_value(self, value: VariableValue) -> bool:
        return value in self.values

    def generate_random(self) -> VariableValue:
        if self.values:
            return random.choice(tuple(self.values))
        raise EvaluationException('Can not choose a random element from empty domain.')

    def pretty_print(self, indentation: str = '', first_line_indentation: int = 0, max_target_length: int = 120,
                     chain: LightVariableResolverChainLike = LightIdentityVariableResolverChain()) -> str:
        values = self.values
        if not values:
            return '{}'
        values = list(sorted(values))
        res = '{'
        current_indentation = first_line_indentation + 1
        average_length = sum((len(str(v)) for v in values)) / len(values)
        if max_target_length - first_line_indentation > (1.5 * average_length + 10):
            indentation = ' ' * (current_indentation + 2)
        last_index = len(values) - 1
        for i, value in enumerate(values):
            value = str(value)
            if current_indentation + 1 + len(value) + 1 > max_target_length:
                res += '\n' + indentation
                current_indentation = len(indentation)
            else:
                res += ' '
                current_indentation += 1
            res += value
            current_indentation += len(value)
            if i != last_index:
                res += ','
                current_indentation += 1
        return res + ' }'

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, DefinedDomain):
            return False
        return self.values == other.values

    def __hash__(self):
        return sum((hash(v) for v in self.values))


class DefinedNumericDomain(DefinedDomain):
    @property
    def allows_subtraction(self) -> bool:
        return True

    @property
    def allows_division(self) -> bool:
        return True

    @property
    def allows_and(self) -> bool:
        return True

    @property
    def allows_or(self) -> bool:
        return True


def defined_domain_for(values: list[str]) -> Domain:
    def try_cast(v: str) -> str | int | float:
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v

    values = [try_cast(v) for v in values]
    if all((isinstance(v, int) or isinstance(v, float) for v in values)):
        return DefinedNumericDomain(values)
    values = [str(v) for v in values]
    if len(values) == 2:
        v0 = values[0].lower()
        v1 = values[1].lower()
        if (v0 == 'true' and v1 == 'false') or (v0 == 'false' and v1 == 'true'):
            return BooleanDomain()

    return DefinedDomain(values)


# noinspection PyAbstractClass
class KnownDomain(Domain, metaclass=SingletonFromAbc):
    known_domains: 'list[KnownDomain]' = []
    name_to_domain: 'dict[str, KnownDomain]' = {}

    @classmethod
    def find_domain(cls, symbol: str):
        pass

    def __init__(self, symbol: str, *additional_symbols: str):
        self.symbol = symbol
        self.names = {symbol, *additional_symbols}

    def __init_subclass__(cls, **kwargs):
        domain: KnownDomain = cls()
        for name in domain.names:
            other_domain = KnownDomain.name_to_domain.get(name)
            if other_domain is not None:
                raise ValueError(f'Overlapping symbol {name} between {other_domain} and {domain}')
            KnownDomain.name_to_domain[name] = domain
        KnownDomain.known_domains.append(domain)

    def __repr__(self) -> str:
        return self.symbol

    def pretty_print(self, indentation: str = '', first_line_indentation: int = 0, max_target_length: int = 120,
                     chain: LightVariableResolverChainLike = LightIdentityVariableResolverChain()) -> str:
        return self.symbol


# __eq__ and __hash__ are implemented by singleton
# noinspection PyAbstractClass
class RealDomain(KnownDomain):

    def __init__(self):
        super().__init__('R', '\u211D')

    @property
    def allows_subtraction(self) -> bool:
        return True

    @property
    def allows_division(self) -> bool:
        return True

    @property
    def allows_ordering(self) -> bool:
        return True

    @property
    def allows_and(self) -> bool:
        return False

    @property
    def allows_or(self) -> bool:
        return False

    def contains_value(self, value: VariableValue) -> bool:
        return isinstance(value, int) or isinstance(value, float)

    def generate_random(self) -> VariableValue:
        return random.gauss(0, 1)


# __eq__ and __hash__ are implemented by singleton
# noinspection PyAbstractClass
class IntegralDomain(KnownDomain):

    def __init__(self):
        super().__init__('Z', '\u2124')

    @property
    def allows_subtraction(self) -> bool:
        return True

    @property
    def allows_division(self) -> bool:
        return True

    @property
    def allows_ordering(self) -> bool:
        return True

    @property
    def allows_and(self) -> bool:
        return True

    @property
    def allows_or(self) -> bool:
        return True

    def contains_value(self, value: VariableValue) -> bool:
        return isinstance(value, int) or (isinstance(value, float) and value == int(value))

    def generate_random(self) -> VariableValue:
        return random.randint(-1000, 1000)


# __eq__ and __hash__ are implemented by singleton
# noinspection PyAbstractClass
class NaturalDomain(KnownDomain):

    def __init__(self):
        super().__init__('N', '\u2115')

    @property
    def allows_subtraction(self) -> bool:
        return True

    @property
    def allows_division(self) -> bool:
        return True

    @property
    def allows_ordering(self) -> bool:
        return True

    @property
    def allows_and(self) -> bool:
        return True

    @property
    def allows_or(self) -> bool:
        return True

    def contains_value(self, value: VariableValue) -> bool:
        return (isinstance(value, int) and value >= 0) \
               or (isinstance(value, float) and value == int(value) and value >= 0)

    def generate_random(self) -> VariableValue:
        return random.randint(0, 1000)


# __eq__ and __hash__ are implemented by singleton
# noinspection PyAbstractClass
class BooleanDomain(FiniteDomain, KnownDomain):

    def __init__(self):
        super().__init__('B', '\U0001D539')

    @property
    def values(self) -> set[str | int | float | bool]:
        return {True, False}

    @property
    def allows_subtraction(self) -> bool:
        return False

    @property
    def allows_division(self) -> bool:
        return False

    @property
    def allows_ordering(self) -> bool:
        return True

    @property
    def allows_and(self) -> bool:
        return True

    @property
    def allows_or(self) -> bool:
        return True

    def contains_value(self, value: VariableValue) -> bool:
        if isinstance(value, bool):
            return True
        if isinstance(value, str):
            match value.lower():
                case 'true' | 'false':
                    return True
        return False

    def generate_random(self) -> VariableValue:
        return random.random() < 0.5


@dataclass(order=True, slots=True, unsafe_hash=True)
class VariableDeclaration(SourceCode, ParseTreeNode):
    identifier: str = field(hash=True, compare=True)
    domain: Domain = field(compare=False)

    def pretty_print(self, indentation: str = '', first_line_indentation: int = 0, max_target_length: int = 120,
                     chain: LightVariableResolverChainLike = LightIdentityVariableResolverChain(),
                     *, extra_indent_on_line_break: str = '') -> str:
        identifier = chain.get_renamed_full_name(self.identifier)
        if len(identifier) + first_line_indentation + 3 > max_target_length:
            res = f'\n{indentation}{identifier}:'
            indentation += extra_indent_on_line_break
            first_line_indentation = len(indentation) - 1
        else:
            res = f'{identifier}:'
        first_line_indentation += len(identifier) + 1
        return f'{res} {self.domain.pretty_print(indentation, first_line_indentation, max_target_length, chain)}'


@dataclass(order=True, slots=True)
class CompressedVariableDeclaration(ParseTreeNode):
    variables: list[str] = field(hash=True)
    domain: Domain = field(compare=False)

    def __post_init__(self):
        if hasattr(super(CompressedVariableDeclaration, self), '__post_init__'):
            # noinspection PyUnresolvedReferences
            super(CompressedVariableDeclaration, self).__post_init__()
        if len(self.variables) == 0:
            raise ValueError('variables must not be empty')

    def pretty_print(self, indentation: str = '', first_line_indentation: int = 0, max_target_length: int = 120,
                     chain: LightVariableResolverChainLike = LightIdentityVariableResolverChain(),
                     *, force_next_line: bool = False) -> str:
        res = ''
        current_indentation = first_line_indentation
        indentation += '  '
        variables = [chain.get_renamed_full_name(v) for v in self.variables]
        for i, variable in enumerate(variables[:-1]):
            prepend_space = i > 0
            if force_next_line \
                    or current_indentation + (1 if prepend_space else 0) + len(variable) + 1 > max_target_length:
                res += '\n' + indentation
                current_indentation = len(indentation)
            elif prepend_space:
                res += ' '
                current_indentation += 1
            res += variable + ','
            current_indentation += len(variable) + 1
        # Since it is only used for its pretty_print, the location doesn't matter
        # noinspection PyTypeChecker
        end_str = VariableDeclaration(None, variables[-1], self.domain) \
            .pretty_print(indentation, current_indentation + (1 if len(variables) > 1 else 0), max_target_length,
                          chain, extra_indent_on_line_break=' ')
        if res and end_str[0] != '\n':
            res += ' '
        res += end_str
        return res


def find_connection_for_awaits(awaiting_atom: Atom, awaited_atom: Atom) -> Iterable[str]:
    controlled = set(awaited_atom.controlled_variables)
    for variable in awaiting_atom.awaited_variables:
        if variable in controlled:
            yield variable


@dataclass(frozen=True)
class VariableResolverChain(LightVariableResolverChainLike):
    head_resolver: 'ModuleLike'
    sub_resolver: 'VariableResolverChain' = None

    @overload
    def get_renamed(self, original_name: str | ScopedVariable, ) -> ScopedVariable:
        pass

    @overload
    def get_renamed(self, atom: Atom | ScopedAtom) -> ScopedAtom:
        pass

    def get_renamed(self, original: str | ScopedVariable | Atom | ScopedAtom):
        if self.sub_resolver is not None:
            original = self.sub_resolver.get_renamed(original)
        return self.head_resolver.get_renamed(original, self.sub_resolver)

    def get_renamed_full_name(self, original_name: str) -> str:
        return self.get_renamed(original_name).full_name

    def __repr__(self) -> str:
        return '(' + self.head_resolver.name + ')' + \
               ('' if self.sub_resolver is None else '->' + repr(self.sub_resolver))


class Executable(ABC):

    def atoms_in_execution_order(self) -> 'Iterable[tuple[VariableResolverChain, Atom]]':
        res: set[tuple[VariableResolverChain, Atom]] = set()
        for chain, atom, _ in self.controlled_variables_with_atoms_in_execution_order():
            res.add((chain, atom))
        return res

    @abstractmethod
    def controlled_variables_with_atoms_in_execution_order(self) -> 'Iterable[tuple[VariableResolverChain, Atom, str]]':
        pass

    @abstractmethod
    def get_external_variables(self) -> 'Iterable[tuple[ScopedVariable, Domain]]':
        pass


class ModuleStore(dict):

    def __getitem__(self, k: str) -> 'ModuleLike':
        return super().__getitem__(k)

    def __setitem__(self, k: str, v: 'ModuleLike'):
        raise ValueError('Operation not supported. Use add(v)')

    def add(self, module: 'ModuleLike'):
        module_name = module.name
        duplicate_module = self.get(module_name)
        if duplicate_module is not None:
            raise ConsistencyException(f'Got multiple modules with name {module_name} at {duplicate_module.source} and '
                                       f'{module.source}')
        super().__setitem__(module_name, module)


@dataclass
class ProtoModule(ABC):
    _name: str = field(init=False, default=None)

    @abstractmethod
    def dependencies(self) -> Iterable[str]:
        pass

    def set_name(self, name: str):
        assert name
        self._name = name

    @property
    @abstractmethod
    def _derived_name(self) -> str:
        pass

    @property
    def name(self) -> str:
        return self._name or self._derived_name

    @abstractmethod
    def as_module(self, loaded_modules: ModuleStore) -> 'ModuleLike':
        pass

    @abstractmethod
    def pretty_print(self, indentation: str = '', first_line_indentation: int = 0, max_target_length: int = 120,
                     chain: LightVariableResolverChainLike = LightIdentityVariableResolverChain(),
                     is_root: bool = True) -> str:
        pass


@dataclass
class DerivedModule(ProtoModule, SourceCode, ABC):
    pass


def pretty_print_module(module: 'ModuleLike', indentation: str, first_line_indentation: int, max_target_length: int,
                        chain: LightVariableResolverChainLike) -> str:
    match module:
        case Module(name=name):
            if first_line_indentation + len(name) > max_target_length:
                return f'\n{indentation}{name}'
            else:
                return name
        case _:
            return module.pretty_print(indentation, first_line_indentation, max_target_length, chain)


@dataclass(slots=True)
class ModuleReference(ProtoModule, SourceCode):
    main_name: Token

    def dependencies(self) -> Iterable[str]:
        yield self.main_name.token

    def set_name(self, name: str):
        raise ConsistencyException(f'The assigning of a module at {self.source} is not allowed. Only derived modules '
                                   f'can be assigned.')

    @property
    def _derived_name(self) -> str:
        return self.main_name.token

    def as_module(self, loaded_modules) -> 'ModuleLike':
        return loaded_modules[self.main_name.token]

    def pretty_print(self, indentation: str = '', first_line_indentation: int = 0, max_target_length: int = 120,
                     chain: LightVariableResolverChainLike = LightIdentityVariableResolverChain(),
                     is_root: bool = True) -> str:
        assert not is_root
        if first_line_indentation + len(self.name) > max_target_length:
            return '\n' + indentation + self.name
        else:
            return self.name


@dataclass(slots=True, unsafe_hash=True)
class ModuleLike(Executable, SourceCode, ParseTreeNode, ABC):
    name: str = field(hash=True)
    private_variables: list[VariableDeclaration] = field(hash=False, compare=False)
    interface_variables: list[VariableDeclaration] = field(hash=False, compare=False)
    external_variables: list[VariableDeclaration] = field(hash=False, compare=False)
    controlled_variables: dict[str, VariableDeclaration] = field(init=False, hash=False, compare=False)
    observable_variables: dict[str, VariableDeclaration] = field(init=False, hash=False, compare=False)
    module_variables: dict[str, VariableDeclaration] = field(init=False, hash=False, compare=False)

    # noinspection PyTypeChecker
    def __post_init__(self):
        if hasattr(super(ModuleLike, self), '__post_init__'):
            # noinspection PyUnresolvedReferences
            super(ModuleLike, self).__post_init__()
        self.private_variables = sorted(self.private_variables)
        self.interface_variables = sorted(self.interface_variables)
        self.external_variables = sorted(self.external_variables)
        self.controlled_variables = {variable.identifier: variable for variable in
                                     itertools.chain(self.private_variables, self.interface_variables)}
        self.observable_variables = {variable.identifier: variable for variable in
                                     itertools.chain(self.interface_variables, self.external_variables)}
        self.module_variables = {variable.identifier: variable for variable in
                                 itertools.chain(self.private_variables, self.interface_variables,
                                                 self.external_variables)}

    def is_private_variables(self, variable: str) -> bool:
        return any((variable == v.identifier for v in self.private_variables))

    def is_interface_variables(self, variable: str) -> bool:
        return any((variable == v.identifier for v in self.interface_variables))

    def is_external_variables(self, variable: str) -> bool:
        return any((variable == v.identifier for v in self.external_variables))

    @overload
    def get_renamed(self, original_name: str | ScopedVariable,
                    sub_chain: VariableResolverChain | None) -> ScopedVariable:
        pass

    @overload
    def get_renamed(self, atom: Atom | ScopedAtom, sub_chain: VariableResolverChain | None) -> ScopedAtom:
        pass

    @lru_cache
    def get_renamed(self, original: str | ScopedVariable | Atom | ScopedAtom,
                    sub_chain: VariableResolverChain | None):
        match original:
            case str(name) | ScopedVariable(variable=name) if self.is_external_variables(name):
                return ScopedVariable(Qualifier.external, None, name)
            case str(name) | ScopedVariable(variable=name) if self.is_interface_variables(name):
                return ScopedVariable(Qualifier.interface, self.name, name)
            case str(name) | ScopedVariable(variable=name):
                return ScopedVariable(Qualifier.private, self.name, name)
            case Atom():
                return ScopedAtom(self.name, original)
            case ScopedAtom():
                return original.new_scope(self.name)

    @abstractmethod
    def controlled_variables_with_atoms(self) -> Iterable[tuple[Atom, str]]:
        pass

    def get_external_variables(self) -> 'Iterable[tuple[ScopedVariable, Domain]]':
        for variable in self.external_variables:
            yield ScopedVariable(Qualifier.external, None, variable.identifier), variable.domain


@dataclass(slots=True, unsafe_hash=True)
class Module(ModuleLike):
    _atoms: list[Atom] = field(init=False, default=None, hash=False, compare=False)
    _registered_controlled_variables: dict[str, str] = field(init=False, hash=False, compare=False)
    _chain: VariableResolverChain = field(init=False, hash=False, compare=False)

    def __repr__(self):
        return self.name

    def _assert_variable_set(self, variable_sets: Iterable[tuple[str, list[VariableDeclaration]]]):
        unique_variables = {}
        for domain, variables in variable_sets:
            for variable in variables:
                variable_name = variable.identifier
                duplicate = unique_variables.get(variable_name)
                if duplicate:
                    duplicate_domain, duplicate_variable = duplicate
                    raise ValueError(f'Got duplicated variables in module {self.name}: '
                                     f'{duplicate_domain} variable {duplicate_variable} '
                                     f'and {domain} variable {variable}')
                unique_variables[variable_name] = domain, variable

    # noinspection PyTypeChecker
    def __post_init__(self):
        super(Module, self).__post_init__()
        self._assert_variable_set([('private', self.private_variables),
                                   ('interface', self.interface_variables),
                                   ('external', self.external_variables)])
        self._registered_controlled_variables = {}
        self._chain = VariableResolverChain(self)

    def register_atom_controls(self, atom_name: str, variable_name: str):
        if variable_name not in self.controlled_variables:
            raise ValueError(f"Atom {atom_name} in module {self.name} want's to control variable {variable_name} but "
                             f"{variable_name} is not a controllable variable (neither private nor interface) "
                             f"in module {self.name}")
        duplicate = self._registered_controlled_variables.get(variable_name)
        if duplicate:
            raise ValueError(f'Multiple atoms in module {self.name} want to control variable {variable_name}: '
                             f'{duplicate} and {atom_name}')
        self._registered_controlled_variables[variable_name] = atom_name

    def _assert_atom_names(self, atoms: list[Atom]) -> Iterable[ValueError]:
        unique_names = set()
        for atom in atoms:
            name = atom.name
            if name in unique_names:
                yield ValueError(f'Got duplicated atom "{name}" in module {self.name}.')
            unique_names.add(name)

    def _assert_atom_controls_are_initialized(self, atoms: list[Atom]) -> Iterable[ValueError]:
        for atom in atoms:
            for command in atom.init:
                missing = set(atom.controlled_variables)
                for assignment in command.assignments:
                    if assignment.target_name in missing:
                        missing.remove(assignment.target_name)
                if missing:
                    yield ValueError(
                        f'The init guarded assignment at {command.source} of atom {atom.name} in module '
                        f'{self.name} misses to assign the following controlled variables: '
                        f'{", ".join(missing)}')

    def _assert_atom_guarded_expression_does_not_assign_multiple_times(self, atoms: list[Atom]) -> Iterable[ValueError]:
        for atom in atoms:
            for command in atom.all_commands:
                assignments_map = defaultdict(list)
                for assignment in command.assignments:
                    assignments_map[assignment.target_name].append(assignment)
                for variable, assignments in assignments_map.items():
                    if len(assignments) > 1:
                        yield ValueError(
                            f"The guarded assignment at {command.source} of atom {atom.name} in module "
                            f"{self.name} assigns {variable}' multiple times. See: "
                            f"{', '.join((str(a.source) for a in assignments))}")

    def _calculate_atom_order(self, atoms: list[Atom]) -> list[Atom]:
        # find an execution order using depth first search on the graph defined by the awaits relation
        # 1. Determine which variable is bound by which atom
        atoms_lookup: dict[str, Atom] = {atom.name: atom for atom in atoms}
        bound_table: dict[str, Atom] = {variable_name: atoms_lookup[atom_name] for
                                        variable_name, atom_name in
                                        self._registered_controlled_variables.items()}
        # 2. Perform DFS
        sorter = graphlib.TopologicalSorter()
        for atom in atoms:
            sorter.add(atom, *(bound_table.get(await_variable) for await_variable in atom.awaited_variables))

        try:
            return list((a for a in sorter.static_order() if a is not None))
        except graphlib.CycleError as e:
            cycle = e.args[1]
            msg = f'Cyclic dependency in module {self.name}: '
            for awaited, awaiting in zip(cycle, cycle[1:]):
                vs = ', '.join(find_connection_for_awaits(awaiting_atom=awaiting, awaited_atom=awaited))
                msg += f'{awaited.module.name}.{awaited.name}[ {vs} ] was awaited by '
            msg += f'{cycle[-1].module.name}.{cycle[-1].name}.'

            raise ConsistencyException(msg) from None

    def _infer_types(self, atoms: list[Atom]) -> Iterable[TypeException]:
        for atom in atoms:
            for command in atom.all_commands:
                # noinspection PyProtectedMember
                yield from command._infer_types(self)

    @property
    def atoms(self) -> list[Atom]:
        return self._atoms

    @atoms.setter
    def atoms(self, atoms: list[Atom]):
        error = ConsistencyException('Module contains errors. See causes for detail.')
        if self._atoms is not None:
            raise ValueError('atoms can only be set once')
        if len(self._registered_controlled_variables) != len(self.controlled_variables):
            error.add_causes(ValueError(f'Not all controlled variables of module {self.name} where bound: '
                                        + ', '.join(sorted((name for name in self.controlled_variables if
                                                            name not in self._registered_controlled_variables)))))

        error.add_causes(self._assert_atom_names(atoms))
        error.add_causes(self._assert_atom_controls_are_initialized(atoms))
        error.add_causes(self._assert_atom_guarded_expression_does_not_assign_multiple_times(atoms))
        try:
            self._atoms = self._calculate_atom_order(atoms)
        except ConsistencyException as e:
            error.add_causes(e)

        error.add_causes(self._infer_types(atoms))

        if error.has_causes():
            raise error

    @property
    def is_combinatorial(self) -> bool:
        if self._atoms is None:
            raise ValueError('Not initialized')
        return all((a.is_combinatorial for a in self._atoms))

    @property
    def is_sequential(self) -> bool:
        return not self.is_combinatorial

    @property
    def is_passive(self) -> bool:
        if self._atoms is None:
            raise ValueError('Not initialized')
        return all((a.is_passive for a in self._atoms))

    @property
    def is_closed(self) -> bool:
        return not self.is_open

    @property
    def is_open(self) -> bool:
        return len(self.external_variables) > 0

    @property
    def is_finite_module(self) -> bool:
        return all((isinstance(v.domain, FiniteDomain) for v in self.module_variables.values()))

    @staticmethod
    def compress_variable_declaration(variable_declarations: list[VariableDeclaration]) \
            -> list[CompressedVariableDeclaration]:
        groups = defaultdict(list)
        for variable_declaration in variable_declarations:
            groups[variable_declaration.domain].append(variable_declaration.identifier)
        # Seems like the type checker doesn't recognise the order=True of the dataclass
        # noinspection PyTypeChecker
        return sorted([CompressedVariableDeclaration(variables, domain) for domain, variables in groups.items()])

    def pretty_print(self, indentation: str = '', first_line_indentation: int = 0, max_target_length: int = 120,
                     chain: LightVariableResolverChainLike = LightIdentityVariableResolverChain()) -> str:
        res = f'module {self.name} is'
        indentation += '  '
        first_line_indentation = len(indentation)
        for name, variables in [
            ('private', Module.compress_variable_declaration(self.private_variables)),
            ('interface', Module.compress_variable_declaration(self.interface_variables)),
            ('external', Module.compress_variable_declaration(self.external_variables)),
        ]:
            if not variables:
                continue
            res += f'\n{indentation}{name}'
            current_indentation = first_line_indentation + len(name)
            force_next_line = False
            for i, variable in enumerate(variables):
                if i > 0:
                    res += ';'
                    current_indentation += 1
                # +1 for space
                variable_str = variable.pretty_print(indentation, current_indentation + 1, max_target_length,
                                                     chain, force_next_line=force_next_line)
                if variable_str[0] != '\n':
                    res += ' '
                res += variable_str
                force_next_line = isinstance(variable.domain, DefinedDomain) and len(variable.domain.values) > 5
                current_indentation = ParseTreeNode.offset(res, first_line_indentation)

        if self.atoms:
            for atom in self.atoms:
                res += '\n\n' + indentation + atom.pretty_print(indentation, first_line_indentation, max_target_length,
                                                                chain)

        return res

    def atoms_in_execution_order(self) -> 'Iterable[tuple[VariableResolverChain, Atom]]':
        for atom in self.atoms:
            yield self._chain, atom

    def controlled_variables_with_atoms_in_execution_order(self) -> 'Iterable[tuple[VariableResolverChain, Atom, str]]':
        for atom, variable in self.controlled_variables_with_atoms():
            yield self._chain, atom, variable

    def controlled_variables_with_atoms(self) -> Iterable[tuple[Atom, str]]:
        for atom in self.atoms:
            for variable in atom.controlled_variables:
                yield atom, variable


@dataclass(slots=True)
class ProtoParallelModule(DerivedModule):
    lhs: ProtoModule
    rhs: ProtoModule

    def dependencies(self) -> Iterable[str]:
        yield from self.lhs.dependencies()
        yield from self.rhs.dependencies()

    @property
    def _derived_name(self) -> str:
        return f'{self.lhs.name} || {self.rhs.name}'

    def as_module(self, loaded_modules: ModuleStore) -> 'ParallelModule':
        lhs = self.lhs.as_module(loaded_modules)
        rhs = self.rhs.as_module(loaded_modules)
        return ParallelModule(self.source, self.name, lhs, rhs)

    def pretty_print(self, indentation: str = '', first_line_indentation: int = 0, max_target_length: int = 120,
                     chain: LightVariableResolverChainLike = LightIdentityVariableResolverChain(),
                     is_root: bool = True) -> str:
        def pp_impl() -> str:
            nonlocal first_line_indentation
            r = self.lhs.pretty_print(indentation, first_line_indentation, max_target_length, chain, is_root=False)
            first_line_indentation = ParseTreeNode.offset(r, first_line_indentation)
            if first_line_indentation + 3 > max_target_length:
                r += '\n' + indentation + '||'
                first_line_indentation = len(indentation) + 2
            else:
                r += ' ||'
                first_line_indentation += 3
            rhs_str = self.rhs.pretty_print(indentation, first_line_indentation, max_target_length, chain,
                                            is_root=False)
            if rhs_str and rhs_str[0] != '\n':
                r += ' '
            return r + rhs_str

        if is_root:
            if first_line_indentation + len(self.name) + 3 > max_target_length:
                res = '\n' + indentation
                first_line_indentation = len(indentation)
            else:
                res = ''
            res += self.name + ' :='
            first_line_indentation + len(self.name) + 4
            indentation += '  '
            sub_part = pp_impl()
            if sub_part[0] != '\n':
                res += ' '
                indentation = ' ' * ParseTreeNode.offset(res, first_line_indentation)
            return res + sub_part
        else:
            return pp_impl()


@dataclass(slots=True, unsafe_hash=True)
class ParallelModule(ModuleLike):
    lhs: ModuleLike = field(hash=False)
    rhs: ModuleLike = field(hash=False)
    private_variables: list[VariableDeclaration] = field(hash=False, init=False)
    interface_variables: list[VariableDeclaration] = field(hash=False, init=False)
    external_variables: list[VariableDeclaration] = field(hash=False, init=False)
    _atom_order: list[tuple[VariableResolverChain, Atom]] = field(hash=False, init=False)

    def __repr__(self):
        return self.name

    def _calculate_atom_order(self) -> list[tuple[VariableResolverChain, Atom]] | ConsistencyException:
        # find an execution order using depth first search on the graph defined by the awaits relation
        # 1. Get a map from variable to atom name and a map from atom name to atom
        variable_to_atom_name: dict[str, str] = {
            VariableResolverChain(self, chain).get_renamed(variable).variable: chain.get_renamed(atom).full_name
            for chain, atom, variable in itertools.chain(self.lhs.controlled_variables_with_atoms_in_execution_order(),
                                                         self.rhs.controlled_variables_with_atoms_in_execution_order())
        }
        atom_name_to_atom: dict[str, tuple[VariableResolverChain, Atom]] = {
            chain.get_renamed(atom).full_name: (VariableResolverChain(self, chain), atom) for chain, atom in
            itertools.chain(self.lhs.atoms_in_execution_order(), self.rhs.atoms_in_execution_order())
        }
        # 2. Perform DFS on atom names
        sorter = graphlib.TopologicalSorter()

        # Ensure current order is kept
        def add_original_order(atoms: Iterable[tuple[VariableResolverChain, Atom]]):
            nonlocal sorter
            atom_list = [chain.get_renamed(atom).full_name for chain, atom in atoms]
            if len(atom_list) == 1:  # Ensure that sorter knows each atom.
                sorter.add(atom_list[0])
            else:
                for first, second in zip(atom_list, atom_list[1:]):
                    sorter.add(second, first)

        add_original_order(self.lhs.atoms_in_execution_order())
        add_original_order(self.rhs.atoms_in_execution_order())

        def add_order_from_awaits(module: ModuleLike):
            nonlocal sorter
            for chain, atom in module.atoms_in_execution_order():
                self_chain = VariableResolverChain(self, chain)
                depending_atom = chain.get_renamed(atom).full_name
                for original_variable in atom.awaited_variables:
                    variable = self_chain.get_renamed(original_variable).variable
                    depended_atom = variable_to_atom_name.get(variable)
                    if depended_atom:
                        sorter.add(depending_atom, depended_atom)

        add_order_from_awaits(self.lhs)
        add_order_from_awaits(self.rhs)

        try:
            return list((atom_name_to_atom[a] for a in sorter.static_order() if a is not None))
        except graphlib.CycleError as e:
            cycle = e.args[1]
            msg = f'Cyclic dependency in parallel module {self.name}: {" was awaited by ".join(cycle)}'
            return ConsistencyException(msg)

    def __post_init__(self):
        # Set variables according to definition
        self.private_variables = self.lhs.private_variables + self.rhs.private_variables
        self.interface_variables = self.lhs.interface_variables + self.rhs.interface_variables
        interface_variables = set(self.interface_variables)
        self.external_variables = [
            v for v in itertools.chain(self.lhs.external_variables, self.rhs.external_variables)
            if v not in interface_variables
        ]
        super(ParallelModule, self).__post_init__()
        # Check that lhs and rhs are compatible
        incompatible_private_lhs: list[str] = []
        incompatible_private_rhs: list[str] = []
        for variable in self.lhs.private_variables:
            if variable.identifier in self.rhs.module_variables:
                incompatible_private_lhs.append(variable.identifier)
        for variable in self.rhs.private_variables:
            if variable.identifier in self.lhs.module_variables:
                incompatible_private_rhs.append(variable.identifier)
        # Controlled variables must not intersect. For private variables, this is already tested. Therefore, only
        # interface variables are tested
        incompatible_interface = {
                                     v.identifier for v in self.lhs.interface_variables
                                 } & {
                                     v.identifier for v in self.rhs.interface_variables
                                 }

        pass
        # Calculate atom execution order
        atom_order = self._calculate_atom_order()
        if isinstance(atom_order, ConsistencyException):
            raise IncompatibleModulesException(self.source, self.lhs, self.rhs, incompatible_private_lhs,
                                               incompatible_private_rhs, incompatible_interface, atom_order)
        elif incompatible_private_lhs or incompatible_private_rhs or incompatible_interface:
            raise IncompatibleModulesException(self.source, self.lhs, self.rhs, incompatible_private_lhs,
                                               incompatible_private_rhs, incompatible_interface, None)
        self._atom_order = atom_order

    @overload
    def get_renamed(self, original_name: str | ScopedVariable,
                    sub_chain: VariableResolverChain | None) -> ScopedVariable:
        pass

    @overload
    def get_renamed(self, atom: Atom | ScopedAtom, sub_chain: VariableResolverChain | None) -> ScopedAtom:
        pass

    @lru_cache
    def get_renamed(self, original: str | ScopedVariable | Atom | ScopedAtom,
                    sub_chain: VariableResolverChain | None):
        self_name = self.name
        if '||' in self_name:
            self_name = f'({self_name})'
        match original:
            case str(name) | ScopedVariable(variable=name) if self.is_external_variables(name):
                return ScopedVariable(Qualifier.external, None, name)
            case str(name) | ScopedVariable(variable=name) if self.is_interface_variables(name):
                return ScopedVariable(Qualifier.interface, self_name, name)
            case str(name) | ScopedVariable(variable=name):
                return ScopedVariable(Qualifier.private, self_name, name)
            case Atom():
                raise ValueError(f'{type(self).__name__} does not have any own Atoms and therefore can not create a '
                                 f'{ScopedAtom.__name__}')
            case ScopedAtom():
                # determine the decision
                return original.new_scope(self_name, decision='l' if self.lhs == sub_chain.head_resolver else 'r')

    def pretty_print(self, indentation: str = '', first_line_indentation: int = 0, max_target_length: int = 120,
                     chain: LightVariableResolverChainLike = LightIdentityVariableResolverChain()) -> str:
        msg = pretty_print_module(self.lhs, indentation, first_line_indentation, max_target_length, chain)
        first_line_indentation = ParseTreeNode.offset(msg, first_line_indentation)
        if first_line_indentation + 3 > max_target_length:
            msg += '\n' + indentation + ' ||'
            first_line_indentation = len(indentation) + 4
        rhs_str = pretty_print_module(self.rhs, indentation, first_line_indentation, max_target_length, chain)
        assert rhs_str
        if rhs_str[0] != '\n':
            msg += ' '
        msg += rhs_str
        return msg

    def atoms_in_execution_order(self) -> 'Iterable[tuple[VariableResolverChain, Atom]]':
        return self._atom_order

    def controlled_variables_with_atoms_in_execution_order(self) -> 'Iterable[tuple[VariableResolverChain, Atom, str]]':
        for chain, atom in self._atom_order:
            for variable in atom.controlled_variables:
                yield chain, atom, variable

    def controlled_variables_with_atoms(self) -> Iterable[tuple[Atom, str]]:
        yield from self.lhs.controlled_variables_with_atoms()
        yield from self.rhs.controlled_variables_with_atoms()


class Renaming(NamedTuple):
    old_name: Token
    new_name: Token


@dataclass(slots=True)
class RenameContext(SourceCode):
    _to_new_name: dict[str, Token] = field(init=False)
    _to_old_name: dict[str, Token] = field(init=False)

    def __init__(self, source: SourceDescriptor, renamings: list[Renaming]):
        self.source = source
        self._to_new_name: dict[str, Token] = {}
        self._to_old_name: dict[str, Token] = {}

        errors = []
        for k, values in itertools.groupby(renamings, key=lambda r: r.old_name.token):
            values = list(values)
            if len(values) > 1:
                errors.append(ConsistencyException(f'Renaming is not right-unique. Variable {k} is renamed to multiple '
                                                   f'other names: {" and ".join(map(str, values))}'))
            else:
                self._to_new_name[k] = values[0].new_name

        for k, values in itertools.groupby(renamings, key=lambda r: r.new_name.token):
            values = list(values)
            if len(values) > 1:
                errors.append(ConsistencyException(f'Renaming is not left-unique. Variable {k} should be the new name '
                                                   f'of multiple variables: {" and ".join(map(str, values))}'))
            else:
                self._to_old_name[k] = values[0].old_name
        if errors:
            raise ConsistencyException('Renaming was not a bijection', errors)

    def merge_with_sub_context(self, sub_context: 'RenameContext') -> 'RenameContext':
        renamings = []
        for center_name in sub_context._to_old_name | self._to_new_name:
            old_from_sub = sub_context._to_old_name.get(center_name)
            new_from_self = self._to_new_name.get(center_name)
            if not old_from_sub:
                # Only in self. Get the token from there.
                old_from_sub = self._to_old_name[new_from_self.token]
            elif not new_from_self:
                # Only in sub. Get the token from there.
                new_from_self = sub_context._to_new_name[old_from_sub.token]
            renamings.append(Renaming(old_from_sub, new_from_self))
        return RenameContext(SourceDescriptor(start=sub_context, end=self), renamings)

    def to_new(self, name: str) -> str:
        new_name = self._to_new_name.get(name)
        return new_name.token if new_name else name

    def to_old(self, name: str) -> str:
        old_name = self._to_old_name.get(name)
        return old_name.token if old_name else name

    def old_variable(self) -> Iterable[Token]:
        yield from self._to_old_name.values()

    def new_variable(self) -> Iterable[Token]:
        yield from self._to_new_name.values()

    def __repr__(self) -> str:
        return self.pretty_print(max_target_length=1000000)

    def pretty_print(self, indentation: str = '', first_line_indentation: int = 0, max_target_length: int = 120) -> str:
        if first_line_indentation < max_target_length:
            res = '['
            first_line_indentation += 1
        else:
            res = '\n' + indentation + '['
            first_line_indentation = len(indentation) + 1
        for i, (new_name, old_name_token) in enumerate(self._to_new_name.items()):
            part = f'{new_name} := {old_name_token.token}{"," if i < len(self._to_new_name) - 1 else ""}'
            if len(part) + first_line_indentation >= max_target_length:
                res += '\n' + indentation
                first_line_indentation = len(indentation)
            elif i > 0:
                res += ' '
                first_line_indentation += 1
            res += part
            first_line_indentation += len(part)
        if first_line_indentation < max_target_length:
            res += ']'
        else:
            res += '\n' + indentation[2:] + ']'
        return res


@dataclass(slots=True)
class ProtoRenamingModule(DerivedModule):
    base: ProtoModule
    renaming_context: RenameContext

    def dependencies(self) -> Iterable[str]:
        return self.base.dependencies()

    @property
    def _derived_name(self) -> str:
        return f'{self.base.name}{self.renaming_context!r}'

    def as_module(self, loaded_modules: ModuleStore) -> 'ModuleLike':
        base = self.base.as_module(loaded_modules)
        return RenamingModule(self.source, self.name, base, self.renaming_context)

    def pretty_print(self, indentation: str = '', first_line_indentation: int = 0, max_target_length: int = 120,
                     chain: LightVariableResolverChainLike = LightIdentityVariableResolverChain(),
                     is_root: bool = True) -> str:
        def pp_impl() -> str:
            nonlocal indentation, first_line_indentation
            match self.base:
                case ProtoRenamingModule() | ModuleReference():
                    braced = False
                case _:
                    braced = True
            if braced:
                if first_line_indentation >= max_target_length:
                    r = '\n' + indentation
                    first_line_indentation = len(indentation)
                else:
                    r = ''
                r += '('
                first_line_indentation += 1
            else:
                r = ''
            r += self.base.pretty_print(indentation, first_line_indentation, max_target_length, chain, is_root=False)
            first_line_indentation = ParseTreeNode.offset(r, first_line_indentation)
            if braced:
                if first_line_indentation >= max_target_length:
                    r += '\n' + indentation
                    first_line_indentation = len(indentation)
                r += ')'
                first_line_indentation += 1
            indentation += '  '
            return r + self.renaming_context.pretty_print(indentation, first_line_indentation, max_target_length)

        if is_root:
            if first_line_indentation + len(self.name) + 3 > max_target_length:
                res = '\n' + indentation
                first_line_indentation = len(indentation)
            else:
                res = ''
            res += self.name + ' :='
            first_line_indentation + len(self.name) + 4
            indentation += '  '
            sub_part = pp_impl()
            if sub_part[0] != '\n':
                res += ' '
                indentation = ' ' * ParseTreeNode.offset(res, first_line_indentation)
            return res + sub_part
        else:
            return pp_impl()


@dataclass(slots=True, unsafe_hash=True)
class RenamingModule(ModuleLike):
    base: ModuleLike = field(hash=False)
    renaming_context: RenameContext = field(hash=False)
    private_variables: list[VariableDeclaration] = field(hash=False, init=False)
    interface_variables: list[VariableDeclaration] = field(hash=False, init=False)
    external_variables: list[VariableDeclaration] = field(hash=False, init=False)

    def __repr__(self):
        return self.name

    def __post_init__(self):
        # All old variables in rename-context must be variables in base
        errors = []
        unknown_variables: list[Token] = []
        blocked_names = set(self.base.module_variables)
        for variable_token in self.renaming_context.old_variable():
            if variable_token.token not in self.base.module_variables:
                unknown_variables.append(variable_token)
            else:
                blocked_names.discard(variable_token.token)

        prefix = f"Renaming of module {self.name} at {self.source} want's to rename "
        unknown_suffix = f'. Variables must be known inside the module'
        if self.base.module_variables:
            unknown_suffix += f'. Variables variables are: {", ".join(self.base.module_variables)}.'
        else:
            unknown_suffix += f' but {self.name} does not have any variables.'
        match unknown_variables:
            case [unknown_variable]:
                errors.append(ConsistencyException(
                    f"{prefix}the unknown variable {unknown_variable}{unknown_suffix}"))
            case [_, *_]:
                errors.append(ConsistencyException(
                    f"{prefix}the following unknown variables: "
                    f"{', '.join(map(str, unknown_variables))}{unknown_suffix}"))

        not_free_variables: list[Token] = []
        # All new variables are not variables of blocked_names (base - old_variables)
        for variable_token in self.renaming_context.new_variable():
            if variable_token.token in blocked_names:
                not_free_variables.append(variable_token)
        match not_free_variables:
            case [not_free_variable]:
                errors.append(ConsistencyException(f'{prefix}to an already in use variable: {not_free_variable}.'))
            case [_, *_]:
                errors.append(ConsistencyException(f'{prefix}to already in use variables: '
                                                   f'{", ".join(map(str, not_free_variables))}.'))
        if errors:
            raise ConsistencyException('Got variable problems in renaming', errors)

        # Set variables according to definition
        def rename(variables: list[VariableDeclaration]) -> Iterable[VariableDeclaration]:
            for variable in variables:
                try:
                    new = self.renaming_context.to_new(variable.identifier)
                except KeyError:
                    yield variable
                else:
                    yield VariableDeclaration(variable.source, new, variable.domain)

        self.private_variables = list(rename(self.base.private_variables))
        self.interface_variables = list(rename(self.base.interface_variables))
        self.external_variables = list(rename(self.base.external_variables))
        super(RenamingModule, self).__post_init__()

    def controlled_variables_with_atoms(self) -> Iterable[tuple[Atom, str]]:
        return self.base.controlled_variables_with_atoms()

    def atoms_in_execution_order(self) -> 'Iterable[tuple[VariableResolverChain, Atom]]':
        for chain, atom in self.base.atoms_in_execution_order():
            yield VariableResolverChain(self, chain), atom

    def controlled_variables_with_atoms_in_execution_order(self) -> 'Iterable[tuple[VariableResolverChain, Atom, str]]':
        for chain, atom, variable in self.base.controlled_variables_with_atoms_in_execution_order():
            yield VariableResolverChain(self, chain), atom, variable

    def pretty_print(self, indentation: str = '', first_line_indentation: int = 0, max_target_length: int = 120,
                     chain: LightVariableResolverChainLike = LightIdentityVariableResolverChain()) -> str:
        if first_line_indentation + len(self.base.name) >= max_target_length:
            res = '\n' + indentation + self.base.name
            first_line_indentation = len(indentation)
        else:
            res = self.base.name
        first_line_indentation += len(self.base.name)
        res += self.renaming_context.pretty_print(indentation + '  ', first_line_indentation, max_target_length)
        return res

    @overload
    def get_renamed(self, original_name: str | ScopedVariable,
                    sub_chain: VariableResolverChain | None) -> ScopedVariable:
        pass

    @overload
    def get_renamed(self, atom: Atom | ScopedAtom, sub_chain: VariableResolverChain | None) -> ScopedAtom:
        pass

    @lru_cache
    def get_renamed(self, original: str | ScopedVariable | Atom | ScopedAtom, sub_chain: VariableResolverChain | None):
        match original:
            case str(name) | ScopedVariable(variable=name) if self.is_external_variables(name):
                return ScopedVariable(Qualifier.external, None, self.renaming_context.to_new(name))
            case str(name) | ScopedVariable(variable=name) if self.is_interface_variables(name):
                return ScopedVariable(Qualifier.interface, self.name, self.renaming_context.to_new(name))
            case str(name) | ScopedVariable(variable=name):
                return ScopedVariable(Qualifier.private, self.name, self.renaming_context.to_new(name))
            case Atom():
                raise ValueError(f'{type(self).__name__} does not have any own Atoms and therefore can not create a '
                                 f'{ScopedAtom.__name__}')
            case ScopedAtom():
                return original.new_scope(self.name)


@dataclass(slots=True)
class ProtoHidingModule(DerivedModule):
    base: ProtoModule
    hidden_variables: list[Token]
    __derived_name: str = field(init=False, hash=False)

    def __post_init__(self):
        self.__derived_name = f'hide {", ".join(sorted({t.token for t in self.hidden_variables}))} in ' \
                              f'({self.base.name})'

    def dependencies(self) -> Iterable[str]:
        return self.base.dependencies()

    @property
    def _derived_name(self) -> str:
        return self.__derived_name

    def as_module(self, loaded_modules: ModuleStore) -> 'HidingModule':
        base = self.base.as_module(loaded_modules)
        return HidingModule(self.source, self.name, base, self.hidden_variables)

    def pretty_print(self, indentation: str = '', first_line_indentation: int = 0, max_target_length: int = 120,
                     chain: LightVariableResolverChainLike = LightIdentityVariableResolverChain(),
                     is_root: bool = True) -> str:
        def pp_impl() -> str:
            nonlocal indentation, first_line_indentation
            match self.base:
                case ProtoHidingModule() | ProtoRenamingModule() | ModuleReference():
                    braced = False
                case _:
                    braced = True
            if first_line_indentation + 4 > max_target_length:
                r = '\n' + indentation + 'hide'
                first_line_indentation = len(indentation) + 4
            else:
                r = 'hide'
                first_line_indentation += 4
            # print the names of the hidden variables.
            variable_list = {t.token for t in self.hidden_variables}
            for i, variable in enumerate(sorted(variable_list)):
                not_last = i < len(variable_list) - 1
                if first_line_indentation + len(variable) + (2 if not_last else 1) > max_target_length:
                    r += '\n' + indentation
                    first_line_indentation = len(indentation)
                else:
                    r += ' '
                    first_line_indentation += 1
                r += variable
                first_line_indentation += len(variable)
                if not_last:
                    r += ','
                    first_line_indentation += 1

            if first_line_indentation + 3 > max_target_length:
                r += '\n' + indentation + 'in'
                first_line_indentation = len(indentation) + 3
            else:
                r += ' in'
                first_line_indentation += 4
            if braced:
                if first_line_indentation >= max_target_length:
                    r += '\n' + indentation + '('
                    first_line_indentation = len(indentation) + 1
                else:
                    r += ' ('
                    first_line_indentation += 1

            r += self.base.pretty_print(indentation + '  ', first_line_indentation, max_target_length, chain,
                                        is_root=False)
            if braced:
                if ParseTreeNode.offset(r, first_line_indentation) >= max_target_length:
                    r += '\n' + indentation + ')'
                else:
                    r += ')'
            return r

        if is_root:
            if first_line_indentation + len(self.name) + 3 > max_target_length:
                res = '\n' + indentation
                first_line_indentation = len(indentation)
            else:
                res = ''
            res += self.name + ' :='
            first_line_indentation += len(self.name) + 4
            indentation += '  '
            sub_part = pp_impl()
            if sub_part[0] != '\n':
                res += ' '
                indentation = ' ' * ParseTreeNode.offset(res, first_line_indentation)
            return res + sub_part
        else:
            return pp_impl()


@dataclass(slots=True, unsafe_hash=True)
class HidingModule(ModuleLike):
    base: ModuleLike = field(hash=False)
    private_variables: list[VariableDeclaration] = field(hash=False, init=False)
    interface_variables: list[VariableDeclaration] = field(hash=False, init=False)
    external_variables: list[VariableDeclaration] = field(hash=False, init=False)

    def __repr__(self):
        return self.name

    def __init__(self, source: SourceDescriptor, name: str, base: ModuleLike, hidden_variables: Iterable[Token]):
        self.source = source
        self.name = name
        self.base = base
        # ensure every variable that should be hidden is an interface variable
        allowed_internal_variables: dict[str, VariableDeclaration] = {
            declaration.identifier: declaration for declaration in self.base.interface_variables
        }
        self.private_variables = list(self.base.private_variables)
        self.external_variables = self.base.external_variables
        unknown_variables: list[Token] = []
        for token in set(hidden_variables):
            try:
                self.private_variables.append(allowed_internal_variables.pop(token.token))
            except KeyError:
                unknown_variables.append(token)
        match unknown_variables:
            case [unknown_variable]:
                raise ConsistencyException(f'Got the variable {unknown_variable} that should be hidden that was not an '
                                           f'internal variable of {self.base.name}.')
            case [_, *_]:
                raise ConsistencyException(f'Got some variables that should be hidden that where not internal variable '
                                           f'of {self.base.name}:{", ".join(map(str, unknown_variables))}')
        self.interface_variables = list(allowed_internal_variables.values())

        self.__post_init__()

    def controlled_variables_with_atoms(self) -> Iterable[tuple[Atom, str]]:
        return self.base.controlled_variables_with_atoms()

    def atoms_in_execution_order(self) -> 'Iterable[tuple[VariableResolverChain, Atom]]':
        for chain, atom in self.base.atoms_in_execution_order():
            yield VariableResolverChain(self, chain), atom

    def controlled_variables_with_atoms_in_execution_order(self) -> 'Iterable[tuple[VariableResolverChain, Atom, str]]':
        for chain, atom, variable in self.base.controlled_variables_with_atoms_in_execution_order():
            yield VariableResolverChain(self, chain), atom, variable

    def pretty_print(self, indentation: str = '', first_line_indentation: int = 0, max_target_length: int = 120,
                     chain: LightVariableResolverChainLike = LightIdentityVariableResolverChain()) -> str:
        # since it is not actually used, max_target_length is ignored
        hidden = {d.identifier for d in self.private_variables} - {d.identifier for d in self.base.private_variables}
        return f'hide {", ".join(sorted(hidden))} in ' + \
               self.base.pretty_print(indentation, first_line_indentation, max_target_length, chain)


@dataclass(slots=True)
class CombinedModule(SourceCode, ParseTreeNode):
    modules: dict[str, Module | ProtoModule]

    def __init__(self, source: SourceDescriptor, modules: Iterable[Module | ProtoModule]):
        self.source = source
        self.modules: dict[str, Module | ProtoModule] = {}
        for module in modules:
            duplicate_module = self.modules.get(module.name)
            if duplicate_module is not None:
                raise ConsistencyException(f'Got multiple modules with name {module.name} at {duplicate_module.source} '
                                           f'and {module.source}')
            self.modules[module.name] = module

    def pretty_print(self, indentation: str = '', first_line_indentation: int = 0, max_target_length: int = 120,
                     chain: LightVariableResolverChainLike = LightIdentityVariableResolverChain()) -> str:
        res = ''
        for module in self.modules.values():
            res += module.pretty_print(indentation, first_line_indentation, max_target_length, chain)
            first_line_indentation = 0
            res += '\n\n'
        return res


@dataclass(slots=True)
class ExecutableCombinedModule(Executable, CombinedModule):
    main_module: ModuleLike

    def __init__(self, source: SourceDescriptor, modules: Iterable[Module | ProtoModule], main_module: ModuleLike):
        super(ExecutableCombinedModule, self).__init__(source, modules)
        self.main_module = main_module

    def pretty_print(self, indentation: str = '', first_line_indentation: int = 0, max_target_length: int = 120,
                     chain: LightVariableResolverChainLike = LightIdentityVariableResolverChain()) -> str:
        modules = super(ExecutableCombinedModule, self).pretty_print(indentation, first_line_indentation,
                                                                     max_target_length, chain)
        return modules + '\n' + indentation + 'run ' + self.main_module.name + '\n'

    def atoms_in_execution_order(self) -> 'Iterable[tuple[VariableResolverChain, Atom]]':
        return self.main_module.atoms_in_execution_order()

    def controlled_variables_with_atoms_in_execution_order(self) -> 'Iterable[tuple[VariableResolverChain, Atom, str]]':
        return self.main_module.controlled_variables_with_atoms_in_execution_order()

    def get_external_variables(self) -> 'Iterable[tuple[ScopedVariable, Domain]]':
        return self.main_module.get_external_variables()
