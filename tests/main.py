from tabulate import tabulate

from reactive_modules.cst import Module, ConsistencyException
from reactive_modules.parser import parse_module
from reactive_modules.simulation import simulate_module, HistoryLogger
from reactive_modules.tokenizer import Tokenizer


def main():
    src = """
            executable Scheduler is
            private prio: { 1, 2 }
            interface task1, task2: N;
                      proc: { 0, 1, 2 };
                      new1, new2: N;

            atom Task1 controls new1
            initupdate
              [] true -> new1' := 5
              [] true -> new1' := 0

            atom Task2 controls new2
            initupdate
              [] true -> new2' := 5
              [] true -> new2' := 0

            atom Executor1 controls task1 reads task1 awaits new1, proc
            init
              [] true -> task1' := new1'
            update
              [] proc' == 1 -> task1' := task1 + new1' - 1 
              [] proc' != 1 -> task1' := task1 + new1' 

            atom Executor2 controls task2 reads task2 awaits new2, proc
            init
              [] true -> task2' := new2'
            update
              [] proc' == 2 -> task2' := task2 + new2' - 1 
              [] proc' != 2 -> task2' := task2 + new2'

            atom Scheduler controls proc, prio reads task1, task2, prio
            init
              [] true -> proc' := 0; prio' := 1
              [] true -> proc' := 0; prio' := 2
            update
              [] task1 == 0 & task2 == 0            -> proc' := 0
              [] prio == 1 & task1 > 0              -> proc' := 1; prio' := 2
              [] prio == 1 & task1 == 0 & task2 > 0 -> proc' := 2
              [] prio == 2 & task2 > 0              -> proc' := 2; prio' := 1
              [] prio == 2 & task2 == 0 & task1 > 0 -> proc' := 1
            """
    try:
        tokenizer = Tokenizer(src)
        module: Module = parse_module(tokenizer)
        if tokenizer.has_more_token():
            raise ValueError('Parse was incomplete')
    except ConsistencyException as e:
        for c in e.all_causes():
            print(c)
        raise

    # simulate_module(executable, command_in=StringIO(''), observer=IOStateLogger(sys.stdout))
    logger = HistoryLogger(log_state_before_micro_round=False)
    simulate_module(module, observer=logger)
    history = list(logger.get_history())
    history = list(map(list, zip(*history)))
    print(tabulate(history, headers='firstrow', tablefmt='html'))


if __name__ == '__main__':
    main()
