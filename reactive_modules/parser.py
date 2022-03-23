import graphlib
import itertools
import string
from dataclasses import dataclass
from functools import wraps, partial
from typing import Callable, TypeVar, Optional, Any, Iterable

from reactive_modules.cst import KnownDomain, Atom, BooleanExpression, TrueFalseExpression, ConstantExpression, \
    ReferenceExpression, BracedBooleanExpression, BracedValueExpression, EqualExpression, \
    NotEqualExpression, LessThanExpression, GreaterThanExpression, LessThanOrEqualExpression, \
    GreaterThanOrEqualExpression, AddExpression, SubtractExpression, MultiplyExpression, DivideExpression, \
    AndExpression, OrExpression, GuardedCommand, Assignment, Expression, Module, VariableDeclaration, Domain, \
    defined_domain_for, SourceDescriptor, ConsistencyException, NotExpression, CombinedModule, \
    ExecutableCombinedModule, ModuleReference, DerivedModule, ModuleLike, ModuleStore, ProtoParallelModule, ProtoModule, \
    RenameContext, ProtoRenamingModule, Renaming, ProtoHidingModule
from reactive_modules.tokenizer import Tokenizer, Token

T = TypeVar('T')

IDENTIFIER_CHARACTERS = set(string.ascii_letters + string.digits + '_')
NUMBER_CHARACTERS = set(string.digits)


class ParseException(Exception):
    def __init__(self, message: str, cause=None):
        super().__init__(message, cause)
        self.message = message
        self.cause = cause

    # noinspection PyMethodMayBeStatic
    def has_causes(self) -> bool:
        return True

    def all_causes(self) -> Iterable[Exception]:
        yield self
        if self.cause is None:
            return
        if hasattr(self.cause, 'has_causes') and hasattr(self.cause, 'all_causes') and self.cause.has_causes():
            yield from self.cause.all_causes()
        else:
            yield self.cause

    def __str__(self):
        return self.message


def with_save_point(func: T) -> T:
    @wraps(func)
    def wrapper(tokenizer: Tokenizer, *args, **kwargs) -> T:
        save_point = tokenizer.save()
        try:
            return func(tokenizer, *args, **kwargs)
        except ParseException:
            tokenizer.restore(save_point)
            raise

    return wrapper


def parse_token(token: str, tokenizer: Tokenizer) -> Token:
    if not tokenizer.has_more_token():
        raise ParseException(f'Expected token "{token}" but reached end of file')
    save_point = tokenizer.save()
    next_token = tokenizer.next_token()
    if next_token.token != token:
        tokenizer.restore(save_point)
        raise ParseException(f'Expected token {token} but got {next_token} instead')
    return next_token


parse_token_semicolon: Callable[[Tokenizer], Token] = partial(parse_token, ';')


def parse_next_token(tokenizer: Tokenizer) -> Token:
    if not tokenizer.has_more_token():
        raise ParseException('Expected token but reached end of file')
    return tokenizer.next_token()


def parse_token_char_pool(allowed_characters: set[str], tokenizer: Tokenizer) -> Token:
    if not tokenizer.has_more_token():
        raise ParseException('Expected token but reached end of file')
    save_point = tokenizer.save()
    next_token = tokenizer.next_token()
    for c in next_token.token:
        if c not in allowed_characters:
            tokenizer.restore(save_point)
            raise ParseException(f'Got unexpected token {next_token}. Token must only contain the following '
                                 f'characters: {"".join(allowed_characters)}')
    return next_token


def is_identifier(identifier: str) -> bool:
    return identifier and identifier[0] not in NUMBER_CHARACTERS


def parse_identifier(tokenizer: Tokenizer) -> Token:
    identifier_token = parse_token_char_pool(IDENTIFIER_CHARACTERS, tokenizer)
    save_point = tokenizer.save()
    if not is_identifier(identifier_token.token):
        tokenizer.restore(save_point)
        raise ParseException(f'{identifier_token} is not a valid identifier.')
    return identifier_token


_last_parse_many_end_exception: ParseException | None = None


def parse_many(parser: Callable[[Tokenizer], T], tokenizer: Tokenizer, *, start: Optional[str] = None,
               separator: Optional[str] = None, end: Optional[str] = None) -> list[T]:
    global _last_parse_many_end_exception
    if start is not None or separator is not None or end is not None:
        save_point = tokenizer.save()
        try:
            return _parse_many_with_token_matching(parser=parser, tokenizer=tokenizer, start=start, separator=separator,
                                                   end=end)
        except ParseException:
            tokenizer.restore(save_point)
            raise
    res = []
    try:
        while tokenizer.has_more_token():
            res.append(parser(tokenizer))
    except ParseException as e:
        _last_parse_many_end_exception = e
        pass
    else:
        _last_parse_many_end_exception = None
    return res


def _parse_many_with_token_matching(parser: Callable[[Tokenizer], T], tokenizer: Tokenizer, *,
                                    start: Optional[str] = None,
                                    separator: Optional[str] = None, end: Optional[str] = None) -> [T]:
    global _last_parse_many_end_exception
    if start is not None:
        parse_token(start, tokenizer)
    res = []
    if end is not None:
        # is it empty?
        save_point = tokenizer.save()
        token = parse_next_token(tokenizer)
        if token.token == end:
            return res
        tokenizer.restore(save_point)
    do_not_catch = False
    try:
        while tokenizer.has_more_token():
            if separator is not None and res:
                save_point = tokenizer.save() if end is None else None
                token = parse_next_token(tokenizer)
                if end is not None:
                    if token.token == end:
                        return res
                    elif token.token != separator:
                        do_not_catch = True
                        raise ParseException(f'Expected either "{separator}" or "{end}" but got {token}')
                elif token.token != separator:
                    tokenizer.restore(save_point)
                    break
            res.append(parser(tokenizer))
    except ParseException as e:
        _last_parse_many_end_exception = e
        if do_not_catch:
            raise
    else:
        _last_parse_many_end_exception = None
    if end is not None:
        parse_token(end, tokenizer)
    return res


def parse_one_of(*parsers: Callable[[Tokenizer], Any], tokenizer: Tokenizer) -> Any:
    if not tokenizer.has_more_token():
        raise ParseException('Expected token but reached end of file')
    errors = []
    for parser in parsers:
        try:
            return parser(tokenizer)
        except ParseException as e:
            errors.append(e)
    save_point = tokenizer.save()
    next_token = tokenizer.next_token()
    tokenizer.restore(save_point)
    raise ParseException(f'All parse attempts failed for token {next_token}', errors)


def parse_maybe(parser: Callable[[Tokenizer], T], tokenizer: Tokenizer) -> Optional[T]:
    try:
        return parser(tokenizer)
    except ParseException:
        return None


def parse_maybe_token(token: str, tokenizer: Tokenizer) -> Optional[T]:
    try:
        return parse_token(token, tokenizer)
    except ParseException:
        return None


def parse_str(tokenizer: Tokenizer) -> str:
    if tokenizer.has_more_token() is None:
        raise ParseException('Expected string literal but reached end of file')
    return tokenizer.next_token().token


@with_save_point
def parse_int(tokenizer: Tokenizer) -> int:
    if tokenizer.has_more_token() is None:
        raise ParseException('Expected int literal but reached end of file')
    token = tokenizer.next_token()
    try:
        return int(token.token)
    except ValueError:
        raise ParseException(f'Expected int literal but got {token}')


@with_save_point
def parse_float(tokenizer: Tokenizer) -> float:
    if tokenizer.has_more_token() is None:
        raise ParseException('Expected float literal but reached end of file')
    token = tokenizer.next_token()
    try:
        return float(token.token)
    except ValueError:
        raise ParseException(f'Expected float literal but got {token}')


def parse_primary_expression(tokenizer: Tokenizer, allowed_variable_references: set[str], section: str,
                             module: Module) -> Expression:
    @with_save_point
    def parse_with_brackets(t: Tokenizer) -> Expression:
        start = parse_token('(', t)
        res = parse_boolean_expression(t, allowed_variable_references, section, module)
        end = parse_token(')', t)
        source = SourceDescriptor(start=start, end=end)
        if isinstance(res, BooleanExpression):
            return BracedBooleanExpression(source, res)
        else:
            return BracedValueExpression(source, res)

    @with_save_point
    def parse_reference(t: Tokenizer) -> ReferenceExpression:
        # If it looks like a known variable, treet it so
        if not tokenizer.has_more_token():
            raise ParseException('Expected reference but reached end of file')
        token = t.next_token()
        variable_name = token.token
        if variable_name and (variable_name[-1] in {'′', '´'}):
            variable_name = variable_name[:-1] + "'"
        refers_to_old = variable_name[-1] != "'"
        name = variable_name if refers_to_old else variable_name[:-1]

        if variable_name in allowed_variable_references:
            return ReferenceExpression(SourceDescriptor(start=token, end=token), name, refers_to_old)
        if name not in module.module_variables:
            raise ParseException(
                f"Unknown variable {variable_name}. Possible options are: {allowed_variable_references}")
        # it looks like a known variable. Determine how it was used wrongly
        if section == 'update':
            if refers_to_old and (variable_name + "'") in allowed_variable_references:
                middle = f"Did you ment to refer to the new value {variable_name}'?"
            elif not refers_to_old and variable_name[:-1] in allowed_variable_references:
                middle = f'Did you ment to refer to the old value {variable_name[:-1]}?'
            else:
                middle = f'You need to {"read" if refers_to_old else "await"} the variable in the atom header.'
        else:
            # was an init or initupdate block
            if (variable_name + "'") in allowed_variable_references:
                middle = f"Did you ment to refer to the new value {variable_name}'?"
            else:
                middle = f'You need to await the variable in the atom header when using it in an init guarded ' \
                         f'expression.'
        end = f' Other options are: {allowed_variable_references}' if allowed_variable_references else ''
        raise ValueError(f'Invalid variable reference {token}. {middle}{end}')

    @with_save_point
    def parse_boolean(t: Tokenizer) -> TrueFalseExpression:
        if not tokenizer.has_more_token():
            raise ParseException('Expected boolean constant but reached end of file')
        token = t.next_token()
        match token.token.lower():
            case 'true':
                return TrueFalseExpression(SourceDescriptor(start=token, end=token), True)
            case 'false':
                return TrueFalseExpression(SourceDescriptor(start=token, end=token), False)
            case _:
                raise ParseException(f'Expected a boolean expression but got {token}')

    @with_save_point
    def parse_value(t: Tokenizer) -> ConstantExpression:
        if not tokenizer.has_more_token():
            raise ParseException('Expected a value but reached end of file')
        token = t.next_token()
        source = SourceDescriptor(start=token, end=token)
        try:
            return ConstantExpression(source, int(token.token))
        except ValueError:
            try:
                return ConstantExpression(source, float(token.token))
            except ValueError:
                return ConstantExpression(source, token.token)

    @with_save_point
    def parse_negation(t: Tokenizer) -> Expression:
        negation = parse_one_of(partial(parse_token, '¬'),
                                partial(parse_token, '!'),
                                tokenizer=t)
        sub_tree = parse_one_of(parse_with_brackets, parse_reference, parse_boolean, parse_value, tokenizer=t)
        return NotExpression(SourceDescriptor(start=negation, end=sub_tree.source), sub_tree)

    return parse_one_of(parse_negation, parse_with_brackets, parse_reference, parse_boolean, parse_value,
                        tokenizer=tokenizer)


T1 = TypeVar('T1')


@dataclass
class Operation:
    op_code: list[str]
    precedence: int
    impl: Callable[[SourceDescriptor, T, T], T1]


BOOLEAN_EXPRESSION_OPERATIONS = [
    Operation(['|', '∨', '\u2228'], 0, OrExpression),
    Operation(['&', '∧', '\u2227'], 1, AndExpression),

    Operation(['=='], 2, EqualExpression),
    Operation(['='], 2, EqualExpression),
    Operation(['!='], 2, NotEqualExpression),
    Operation(['/='], 2, NotEqualExpression),
    Operation(['<'], 3, LessThanExpression),
    Operation(['>'], 3, GreaterThanExpression),
    Operation(['<=', '≤'], 3, LessThanOrEqualExpression),
    Operation(['>=', '≥'], 3, GreaterThanOrEqualExpression),

    Operation(['+'], 4, AddExpression),
    Operation(['-'], 4, SubtractExpression),

    Operation(['*'], 5, MultiplyExpression),
    Operation(['/'], 5, DivideExpression),
]


@with_save_point
def parse_expression(tokenizer: Tokenizer, operations: list[Operation],
                     primary_parser: Callable[[Tokenizer], T]) -> T:
    operations = {c: o for o in operations for c in o.op_code}

    def parse_expression_1(lhs: T, min_precedence: int) -> tuple[T, Token]:
        lookahead = tokenizer.peek()
        lhs_token = lookahead
        while lookahead:
            op = operations.get(lookahead.token)
            source_token = lookahead
            if not op or op.precedence < min_precedence:
                break
            tokenizer.next_token()
            rhs = primary_parser(tokenizer)
            lookahead = tokenizer.peek()
            while lookahead:
                op_ = operations.get(lookahead.token)
                if not op_ or op_.precedence <= op.precedence:
                    break
                rhs, rhs_token = parse_expression_1(rhs, op.precedence + 1)
                lookahead = tokenizer.peek()
            try:
                lhs = op.impl(SourceDescriptor(start=source_token, end=source_token), lhs, rhs)
            except ValueError as e:
                raise ParseException(f'Expected different type {rhs}', e)
        return lhs, lhs_token

    res = parse_expression_1(primary_parser(tokenizer), 0)[0]
    if hasattr(res, 'compress') and callable(res.compress):
        res = res.compress()
    return res


@with_save_point
def parse_boolean_expression(tokenizer: Tokenizer, allowed_variable_references: set[str], section: str,
                             module: Module) -> BooleanExpression | ReferenceExpression:
    first = tokenizer.peek()
    res = parse_expression(tokenizer, BOOLEAN_EXPRESSION_OPERATIONS,
                           partial(parse_primary_expression, allowed_variable_references=allowed_variable_references,
                                   section=section, module=module))
    if not isinstance(res, BooleanExpression) and not isinstance(res, ReferenceExpression):
        raise ParseException(
            f'Expected a boolean expression or a reference to a boolean but got {res} starting at {first}')
    return res


@with_save_point
def parse_assignment(tokenizer: Tokenizer, allowed_assignments: set[str],
                     allowed_variable_references: set[str], section: str, module: Module) -> Assignment:
    if not tokenizer.has_more_token():
        raise ParseException('Expected reference to new value but reached end of file')
    target = tokenizer.next_token()
    parse_token(':=', tokenizer)
    name = target.token
    if name and (name[-1] in {'′', '´'}):
        name = name[:-1] + "'"
    if not target or name not in allowed_assignments:
        msg = f'Can not assign to {target}.'
        if name[-1] != "'":
            msg += f" Did you ment {name}'?"
            if f"{name}'" not in allowed_assignments:
                msg += ' You will need to await the variable in the header too.'
        raise ValueError(msg)
    expression = parse_expression(tokenizer, BOOLEAN_EXPRESSION_OPERATIONS,
                                  partial(parse_primary_expression,
                                          allowed_variable_references=allowed_variable_references,
                                          section=section, module=module))
    return Assignment(SourceDescriptor(start=target, end=tokenizer.last_token), name[:-1], expression)


@with_save_point
def parse_guarded_command(tokenizer: Tokenizer, allowed_assignments: set[str],
                          allowed_variable_references: set[str], section: str, module: Module) -> GuardedCommand:
    start_token = parse_token('[]', tokenizer)
    predicate = parse_boolean_expression(tokenizer, allowed_variable_references, section, module)
    parse_one_of(partial(parse_token, '→'), partial(parse_token, '->'), tokenizer=tokenizer)
    assignments = parse_many(partial(parse_assignment, allowed_assignments=allowed_assignments,
                                     allowed_variable_references=allowed_variable_references, section=section,
                                     module=module),
                             tokenizer, separator=';')
    end = tokenizer.last_token
    return GuardedCommand(SourceDescriptor(start=start_token, end=end), predicate, assignments)


@with_save_point
def parse_init_update(tokenizer: Tokenizer, allowed_assignments: set[str], allowed_init_variable_references: set[str],
                      allowed_update_variable_references: set[str], module: Module) \
        -> tuple[[GuardedCommand], [GuardedCommand]]:
    _parse_init_guarded_command = partial(parse_guarded_command, allowed_assignments=allowed_assignments,
                                          allowed_variable_references=allowed_init_variable_references, section='init',
                                          module=module)
    _parse__update_guarded_command = partial(parse_guarded_command, allowed_assignments=allowed_assignments,
                                             allowed_variable_references=allowed_update_variable_references,
                                             section='update', module=module)
    if not tokenizer.has_more_token():
        raise ParseException('Expected init or initupdate but reached end of file')
    first = tokenizer.next_token()
    if first.token == 'init':
        init = parse_many(_parse_init_guarded_command, tokenizer)
        parse_token('update', tokenizer)
        update = parse_many(_parse__update_guarded_command, tokenizer)
        return init, update
    elif first.token == 'initupdate':
        init_update = parse_many(_parse_init_guarded_command, tokenizer)
        return init_update, init_update
    raise ParseException(f'Expected either an init or initupdate block at {first}.')


@with_save_point
def parse_atom(tokenizer: Tokenizer, module: Module) -> Atom:
    # atom <name> [controls x, y, ...] [reads x, y, ...] [awaits x, y, ...]
    # either:
    # init ...
    # update ...
    # or:
    # initupdate ...
    lazy = parse_maybe(partial(parse_token, 'lazy'), tokenizer) is not None
    start_token = parse_token('atom', tokenizer)
    name_token = parse_identifier(tokenizer)

    @with_save_point
    def parse_labeled_value_list(t: Tokenizer, label: str) -> [Token]:
        if parse_maybe(partial(parse_token, label), tokenizer) is not None:
            values = parse_many(parse_identifier, t, separator=',')
            if not values:
                raise ParseException(f'Expected at least one identifier after {label} in atom definition')
            return values
        else:
            return []

    controls_tokens = parse_labeled_value_list(tokenizer, 'controls')
    controls = {token.token: token for token in controls_tokens}
    unknown_controlled_variables = [token for token in controls_tokens if token.token not in module.module_variables]
    if unknown_controlled_variables:
        raise ValueError(f"Atom {name_token} want's to control at least one variable that is not declared in module "
                         f"{module.name}. Namely: {', '.join((str(t) for t in unknown_controlled_variables))}")

    reads_tokens = parse_labeled_value_list(tokenizer, 'reads')
    unknown_read_variables = [token for token in reads_tokens if token.token not in module.module_variables]
    if unknown_read_variables:
        raise ValueError(f"Atom {name_token} want's to read at least one variable that is not declared in module "
                         f"{module.name}. Namely: {', '.join((str(t) for t in unknown_read_variables))}")

    awaits_tokens = parse_labeled_value_list(tokenizer, 'awaits')
    unknown_awaits_variables = [token for token in awaits_tokens if token.token not in module.module_variables]
    if unknown_awaits_variables:
        raise ValueError(f"Atom {name_token} want's to await at least one variable that is not declared in module "
                         f"{module.name}. Namely: {', '.join((str(t) for t in unknown_awaits_variables))}")
    for token in awaits_tokens:
        c = controls.get(token.token)
        if c:
            raise ValueError(
                f"Atom {name_token.token} want's to await {token} but declared that it controls {c.token} at {c}")

    allowed_assignments: set[str] = {f"{c}'" for c in controls}
    allowed_init_variable_references = {f"{token.token}'" for token in awaits_tokens}
    allowed_update_variable_references = {token.token for token in reads_tokens} | allowed_init_variable_references
    init, update = parse_init_update(tokenizer, allowed_assignments=allowed_assignments,
                                     allowed_init_variable_references=allowed_init_variable_references,
                                     allowed_update_variable_references=allowed_update_variable_references,
                                     module=module)
    try:
        return Atom(SourceDescriptor(start=start_token, end=tokenizer.last_token), name_token.token, module, lazy,
                    [token.token for token in controls_tokens],
                    [token.token for token in reads_tokens],
                    [token.token for token in awaits_tokens], init, update)
    except Exception:
        raise ConsistencyException(f'Construction of atom {name_token.token} failed')


@with_save_point
def parse_known_domain(tokenizer: Tokenizer, domain: KnownDomain) -> Domain:
    if tokenizer.has_more_token() is None:
        raise ParseException('Expected a domain name but reached end of file')
    name_token = tokenizer.next_token()
    if name_token.token not in domain.names:
        raise ParseException(f'Expected one of the following domain names {domain.names} but got {name_token}')
    return domain


@with_save_point
def parse_defined_domain(tokenizer: Tokenizer) -> Domain:
    values = parse_many(parse_str, tokenizer, start='{', separator=',', end='}')
    return defined_domain_for(values)


def parse_domain(tokenizer: Tokenizer) -> Domain:
    if tokenizer.has_more_token() is None:
        raise ParseException('Expected a domain name or a custom domain but reached end of file')
    return parse_one_of(*(partial(parse_known_domain, domain=domain) for domain in KnownDomain.known_domains),
                        parse_defined_domain,
                        tokenizer=tokenizer)


@with_save_point
def parse_variable_declaration(tokenizer: Tokenizer) -> list[VariableDeclaration]:
    identifiers = parse_many(parse_identifier, tokenizer, separator=',', end=':')
    if not identifiers:
        n = tokenizer.peek()
        if n:
            raise ParseException(f'Expected an identifier at {n}')
        else:
            raise ParseException(f'Expected an identifier but reached end of file')
    domain = parse_domain(tokenizer)
    return [
        VariableDeclaration(SourceDescriptor(start=identifiers[0], end=tokenizer.last_token), identifier.token, domain)
        for
        identifier in identifiers]


@with_save_point
def parse_module(tokenizer: Tokenizer) -> Module:
    # module <name> is
    # [private :variable_declaration]
    # [interface :variable_declaration]
    # [external :variable_declaration]
    # [:atom ...]
    start_token = parse_token('module', tokenizer)
    name_token = parse_identifier(tokenizer)
    parse_token('is', tokenizer)

    @with_save_point
    def parse_labeled_variable_declaration(t: Tokenizer, label: str) -> list[VariableDeclaration]:
        if parse_maybe(partial(parse_token, label), tokenizer) is not None:
            values = [v for vs in parse_many(parse_variable_declaration, t, separator=';') for v in vs]
            if not values:
                error_msg = f'Expected at least one variable declaration after {label} '
                if tokenizer.has_more_token():
                    peek = tokenizer.peek()
                    error_msg += f'at {peek.line}:{peek.start} in module definition.'
                else:
                    error_msg += 'but reached end of file.'
                raise ParseException(error_msg)
            return values
        else:
            return []

    private_variables = parse_labeled_variable_declaration(tokenizer, 'private')
    interface_variables = parse_labeled_variable_declaration(tokenizer, 'interface')
    external_variables = parse_labeled_variable_declaration(tokenizer, 'external')

    module = None
    atoms = None
    try:
        source = SourceDescriptor(start=start_token)
        module = Module(source, name_token.token, private_variables, interface_variables, external_variables)
        atoms = parse_many(partial(parse_atom, module=module), tokenizer)
        for atom in atoms:
            for variable_name in atom.controlled_variables:
                module.register_atom_controls(atom.name, variable_name)

        source.set_end(tokenizer.last_token)
        module.atoms = atoms
        return module
    except (ValueError, ParseException, ConsistencyException) as e:
        msg = f'Construction of module {name_token.token} failed. Module is:\n'
        if module:
            msg += module.pretty_print()
        else:
            msg += 'Failed to create module'
        if atoms:
            msg += '\n\nAtoms are:'
            for atom in atoms:
                msg += '\n' + atom.pretty_print()
        else:
            msg += '\n\nNo atoms read'
        raise ConsistencyException(msg, e)


MODULE_OPERATIONS = [
    Operation(['||', '|', '∥'], 0, ProtoParallelModule),
]


@with_save_point
def parse_renaming(tokenizer: Tokenizer) -> RenameContext:
    """
    "[" ,.renaming* "]"
    renaming         := module-reference ":=" <identifier>
    """

    @with_save_point
    def parse_single(t: Tokenizer) -> Renaming:
        old_name = parse_identifier(t)
        parse_token(':=', t)
        new_name = parse_identifier(t)
        return Renaming(old_name, new_name)

    if not tokenizer.has_more_token():
        raise ParseException('Expected [ as the start of a renaming expression but reached end of file.')
    start = tokenizer.peek()
    renamings = parse_many(parse_single, tokenizer, start='[', separator=',', end=']')
    end = tokenizer.last_token
    return RenameContext(SourceDescriptor(start=start, end=end), renamings)


def parse_primary_proto_module(tokenizer: Tokenizer) -> DerivedModule | ModuleReference:
    # @with_save_point is not needed, since only one parse function is used that do restore on exception.
    def parse_module_reference(t: Tokenizer) -> ModuleReference:
        """module-reference := <identifier>"""
        identifier = parse_identifier(t)
        return ModuleReference(SourceDescriptor(start=identifier, end=identifier), identifier)

    @with_save_point
    def parse_module_renaming(t: Tokenizer) -> ProtoRenamingModule:
        """
        renaming-module  := renaming-module "[" ,.renaming* "]"
                          | module-reference "[" ,.renaming* "]"
                          | braced "[" ,.renaming* "]"
        renaming         := module-reference ":=" <identifier>
        """
        base = parse_one_of(parse_module_reference, parse_braced, tokenizer=t)
        renamings = parse_many(parse_renaming, t)
        if not renamings:
            raise ParseException(f'Expected a renaming expression at {t.peek()}')
        renaming = renamings[0]
        for r in renamings[1:]:
            renaming = r.merge_with_sub_context(renaming)
        return ProtoRenamingModule(SourceDescriptor(start=base, end=renaming), base, renaming)

    def parse_braced(t: Tokenizer) -> ProtoModule:
        """
        braced := "(" proto-module ")"
        """
        parse_token("(", t)
        res = parse_proto_module(t)
        parse_token(")", t)
        return res

    def parse_hiding_module(t: Tokenizer) -> ProtoHidingModule:
        """
        hiding-module := "hide" ,.<identifier>+ "in" module-reference
                       | "hide" ,.<identifier>+ "in" renaming-module
                       | "hide" ,.<identifier>+ "in" hiding-module
                       | "hide" ,.<identifier>+ "in" braced
        """
        start = parse_token('hide', t)
        identifiers = parse_many(parse_identifier, t, separator=',')
        if not identifiers:
            raise ParseException(f'Expected at least one identifier at {t.peek()}')
        parse_token('in', t)
        base = parse_primary_proto_module(t)
        return ProtoHidingModule(SourceDescriptor(start=start, end=base), base, identifiers)

    return parse_one_of(parse_hiding_module,
                        parse_module_renaming,
                        parse_braced,
                        parse_module_reference,
                        tokenizer=tokenizer)


def parse_proto_module(tokenizer: Tokenizer) -> ProtoModule:
    """
    proto-module     := hiding-module | renaming-module | parallel-module | braced | module-reference
    braced           := "(" proto-module ")"
    module-reference := <identifier>
    renaming-module  := renaming-module "[" ,.renaming* "]"
                      | module-reference "[" ,.renaming* "]"
                      | braced "[" ,.renaming* "]"
    renaming         := module-reference ":=" <identifier>
    hiding-module    := "hide" ,.<identifier>+ "in" hiding-module
                      | "hide" ,.<identifier>+ "in" renaming-module
                      | "hide" ,.<identifier>+ "in" module-reference
                      | "hide" ,.<identifier>+ "in" braced
    parallel-module  := proto-module "||" proto-module
    """
    return parse_expression(tokenizer, MODULE_OPERATIONS, parse_primary_proto_module)


@with_save_point
def parse_module_like_declaration(tokenizer: Tokenizer) -> ProtoModule | Module:
    if not tokenizer.has_more_token():
        raise ParseException('Expected either module or an identifier')
    start = tokenizer.peek()
    if start.token == 'module':
        module = parse_maybe(parse_module, tokenizer)
        if module:
            return module
    new_name = parse_identifier(tokenizer)
    parse_token(':=', tokenizer)
    proto_module = parse_proto_module(tokenizer)
    try:
        proto_module.set_name(new_name.token)
    except ConsistencyException as e:
        raise ParseException(e.message) from None
    return proto_module


@with_save_point
def parse_run(tokenizer: Tokenizer) -> ProtoModule:
    if not tokenizer.has_more_token():
        raise ParseException("Expected run statement but reached end of file")
    start = tokenizer.next_token()
    if start.token != 'run' and is_identifier(start.token):
        additional_exception = _last_parse_many_end_exception
    else:
        additional_exception = None
    try:
        if start.token != 'run':
            raise ParseException(f'Expected token run but got {start} instead')
        return parse_proto_module(tokenizer)
    except ParseException as e:
        if additional_exception is not None:
            raise ConsistencyException('Maybe an error in module deriving', e) from None
        else:
            raise


def assert_module_has_an_atom(module: ModuleLike):
    try:
        next(iter(module.atoms_in_execution_order()))
    except StopIteration:
        raise ConsistencyException(f'Can not execute module {module.name} with no atoms.')


def parse(tokenizer: Tokenizer, force_executable: bool = False) -> Optional[CombinedModule]:
    def get_parse_exception():
        if _last_parse_many_end_exception is not None:
            return _last_parse_many_end_exception
        unparsed_start = tokenizer.next_token()
        return ParseException(f'Failed to parse entire file. Parser stopped at {unparsed_start.line}:'
                              f'{unparsed_start.start}')

    if tokenizer.has_more_token():
        start = tokenizer.peek()
        try:
            modules = parse_many(parse_module_like_declaration, tokenizer)
            if tokenizer.has_more_token():
                main = parse_run(tokenizer)
                modules += parse_many(parse_module_like_declaration, tokenizer)
            else:
                main = None
            if not tokenizer.has_more_token():
                # All modules and combined modules where read. Now we need to evaluate all combined modules.
                # 1. Determine if there are modules with the same name.
                errors = []
                all_modules: dict[str, Module | DerivedModule] = {}
                for k, duplicates in itertools.groupby(modules, key=lambda m: m.name):
                    duplicates = list(duplicates)
                    if len(duplicates) > 1:
                        errors.append(ConsistencyException(f'Got multiple modules with name {k}: '
                                                           f'{", ".join((str(m.source) for m in duplicates))}.'))
                    else:
                        all_modules[k] = duplicates[0]
                if errors:
                    raise ConsistencyException(f'Name conflict{"s" if len(errors) > 1 else ""}', errors)
                # 2. Prepare dfs to load combined modules
                sorter = graphlib.TopologicalSorter()
                loaded_modules = ModuleStore()
                for module in modules:
                    match module:
                        case Module():
                            loaded_modules.add(module)
                        case DerivedModule():
                            dependencies = list(module.dependencies())
                            for dependency in dependencies:
                                if dependency not in all_modules:
                                    errors.append(ConsistencyException(f'Unknown module "{dependency}" in dependencies '
                                                                       f'of {module.name} ({module.source})'))
                            sorter.add(module.name, *dependencies)
                        case _:
                            assert False, f'Did not expect something like {module}'
                if errors:
                    exception = ConsistencyException(f'Dependency conflict{"s" if len(errors) > 1 else ""}')
                    exception.add_causes(exception)
                    raise exception
                # 3. perform dfs
                sorter.prepare()
                while sorter.is_active():
                    for module_name in sorter.get_ready():
                        module = all_modules[module_name]
                        if isinstance(module, DerivedModule):
                            loaded_modules.add(module.as_module(loaded_modules))
                        sorter.done(module_name)

                source = SourceDescriptor(start=start, end=tokenizer.last_token)
                if main is not None:
                    dependencies = list(main.dependencies())
                    for dependency in dependencies:
                        if dependency not in all_modules:
                            errors.append(ConsistencyException(f'Unknown module "{dependency}" in dependencies '
                                                               f'of run statement.'))
                    if errors:
                        exception = ConsistencyException(f'Dependency conflict{"s" if len(errors) > 1 else ""} in run.')
                        exception.add_causes(errors)
                        raise exception
                    main_module = main.as_module(loaded_modules)
                    assert_module_has_an_atom(main_module)
                    return ExecutableCombinedModule(source, modules, main_module)
                elif force_executable:
                    match modules:
                        case []:
                            raise ConsistencyException('No module found')
                        case [module]:
                            assert_module_has_an_atom(module)
                            return ExecutableCombinedModule(source, [module], module)
                        case modules:
                            raise ConsistencyException(f'Implicit execution is only possible if there is exactly one '
                                                       f'module but found {", ".join((m.name for m in modules))}.')
                else:
                    return CombinedModule(source, modules)
        except ConsistencyException as e:
            if tokenizer.has_more_token():
                e.add_causes(get_parse_exception())
            raise
    if tokenizer.has_more_token():
        raise get_parse_exception()
    return None
