import argparse
import sys
import typing
from dataclasses import dataclass
from io import TextIOWrapper
from typing import TextIO

from reactive_modules.formatting import format_str, parse_file
from reactive_modules.simulation import IOCsvHistoryLogger, NRoundsController, OutputFormat, SimulationStateObserver
from reactive_modules.simulation import simulate_module, InteractiveController, ObserverGroup

T = typing.TypeVar('T')


@dataclass(slots=True)
class AdvancedInteractiveController(InteractiveController):
    def __post_init__(self):
        self.command_out.write(f'''Pressing ctrl + c will end the simulation

Press any key to start the simulation.
''')
        try:
            self.command_in.read(1)
        except KeyboardInterrupt:
            sys.exit(0)


def get_output_from_input(i: TextIOWrapper) -> TextIO:
    # hack for debugger stdin
    if hasattr(i, 'original_stdin'):
        i = i.original_stdin
    match i:
        case TextIOWrapper(name='<stdin>'):
            return sys.stdout
        case TextIOWrapper(name=name, encoding=encoding):
            return open(name, 'w', encoding=encoding)
        case _:
            raise ValueError(f'Failed to get output for input {i}')


def get_name(i: TextIOWrapper) -> str:
    # hack for debugger stdin
    if hasattr(i, 'original_stdin'):
        i = i.original_stdin
    match i:
        case TextIOWrapper(name=name):
            return name
        case _:
            return '?'


def close_io(i: TextIOWrapper | TextIO):
    # hack for debugger stdin
    if hasattr(i, 'original_stdin'):
        i = i.original_stdin
    if i.closed:
        return
    match i:
        case TextIOWrapper(name='<stdin>' | '<stdout>'):
            pass
        case TextIOWrapper():
            i.close()


def format_code(namespace: argparse.Namespace) -> int:
    def get_output(i):
        return get_output_from_input(i) if namespace.output is None else namespace.output

    res = 0
    # noinspection PyShadowingBuiltins
    for input in namespace.input:
        source = get_name(input)
        if source != '<stdin>' and not source.endswith('.rm') and not namespace.force:
            print(f'Skipping file {source} since it is not a .rm file. Use the --force option to force formatting.',
                  file=sys.stderr)
            continue
        content = input.read()
        close_io(input)
        match format_str(content, source, namespace.max_columns):
            case list(errors):
                for error in errors:
                    print(error, file=sys.stderr)
                res = -1
            case data:
                o = get_output(input)
                o.write(data)
                close_io(o)
    return res


def check(namespace: argparse.Namespace) -> int:
    # noinspection PyShadowingBuiltins
    for input in namespace.input:
        source = get_name(input)
        if source != '<stdin>' and not source.endswith('.rm') and not namespace.force:
            print(f'Warning: {source} is not a .rm file. Use the --force option to hide this warning.', file=sys.stderr)
        content = input.read()
        close_io(input)
        match parse_file(content, source):
            case list(errors):
                for error in errors:
                    print(error)  # not stderr since this is expected
    return 0


def run(namespace: argparse.Namespace) -> int:
    if namespace.round_count is None:
        pass
    source = get_name(namespace.source)
    content = namespace.source.read()
    close_io(namespace.source)
    match parse_file(content, source, force_executable=True):
        case list(errors):
            for error in errors:
                print(error, file=sys.stderr)
            return -1
        case module:
            pass

    if namespace.log is not None:
        observer = IOCsvHistoryLogger(log_state_for_micro_round=namespace.verbose, data_out=namespace.log)
    else:
        observer = SimulationStateObserver()

    if namespace.interactive:
        controller = AdvancedInteractiveController(sys.stdin, sys.stdout)
        controller.output_format = namespace.format
        observer = ObserverGroup(observer, controller)
    else:
        controller = NRoundsController(namespace.round_count)

    simulate_module(module, controller=controller, observer=observer)


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='reactive_modules')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enables more verbose output. The correct behavior depends on the sub-command.')
    subparsers = parser.add_subparsers(help='reactive_modules provides two main functionalities: formatting source '
                                            'code and interpreting it. You need to use the individual sub-command for '
                                            'the different tasks.')
    # create the pars for the "a" command
    format_parser = subparsers.add_parser('format', help='Formatting one or more files.')
    format_parser.add_argument('input', nargs='+', type=argparse.FileType('r', encoding='UTF-8'), metavar='INPUT',
                               help='A list of files to format. Pass - to read just from standard in. If output is '
                                    'specified, all content of all input is appended with a line break and then '
                                    'written to output. In verbose mode, the source file name is prepended as a '
                                    'comment. If output is not specified, the files are formatted in place (standard '
                                    'in is written to standard out respectively). If the input file contains '
                                    'syntactical errors, errors are printed to standard error and the content of this '
                                    'file are not written.')
    format_parser.add_argument('-o', '--output', nargs='?', type=argparse.FileType('w', encoding='UTF-8'),
                               metavar='OUTPUT', help='Specifies a single output file grouping all the formatted input '
                                                      'data. If omitted, the output is written into the input files '
                                                      'instead replacing there content.')
    format_parser.add_argument('-f', '--force', action='store_true',
                               help='Enables formatting of arbitrary files not just .rm.')

    class MaxColumnsAction(argparse.Action):

        def __call__(self, pars, namespace, values, option_string=None):
            if not isinstance(values, int):
                pars.error('Maximum columns must be an integer greater than 0.')
            if values < 1:
                pars.error('Maximum columns must be at least 1')

            setattr(namespace, self.dest, values)

    format_parser.add_argument('--max-columns', type=int, action=MaxColumnsAction, default=120,
                               help='Specify how many characters should be at most in one line.')
    format_parser.set_defaults(func=format_code)

    check_parser = subparsers.add_parser('check',
                                         help='Check one or more files for syntactic and some semantic errors.')
    check_parser.add_argument('input', nargs='+', type=argparse.FileType('r', encoding='UTF-8'), metavar='INPUT',
                              help='A list of files to check. Pass - to read just from standard in.')
    check_parser.add_argument('-f', '--force', action='store_true',
                              help='Disable warning for analysing something that is not a .rm file.')
    check_parser.set_defaults(func=check)

    run_parser = subparsers.add_parser('run', help='Run the interpreter on the passed executable.')
    run_parser.add_argument('source', type=argparse.FileType('r', encoding='UTF-8'), metavar='SOURCE',
                            help='File to interpret.')
    mode_group = run_parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('-i', '--interactive', action='store_true', help='Start an interactive simulation session.')
    mode_group.add_argument('-n', '--round-count', type=int, metavar='N',
                            help='Start a simulation for N rounds (counting init as round 0). This means that -n 0 '
                                 'will just run the init round. -n 5 will run the init round and five update rounds. '
                                 'Passing a negative value will result in an endless simulation.')
    run_parser.add_argument('-l', '--log', type=argparse.FileType('w', encoding='UTF-8'), metavar='LOG',
                            help='When specified, the state of the simulation is written into the log file in a csv '
                                 'format. Usually the state is written after each round starting with init. If the '
                                 'verbose flag is set, instead of just logging after each round, after each micro '
                                 'round the state is logged.')
    run_parser.add_argument('-f', '--format', type=str, const='simple', nargs='?', default='simple',
                            choices=[e for e in OutputFormat],
                            help='Only used in interactive mode. It determines the format whenever a table is printed. '
                                 'Defaults to simple')
    run_parser.set_defaults(func=run)

    return parser


def main() -> int:
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()
    if 'func' not in args:
        print(f'You need to select one sub-command.', file=sys.stderr)
        arg_parser.print_usage(file=sys.stderr)
        return 1
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
