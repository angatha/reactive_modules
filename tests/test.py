import unittest

from reactive_modules.cst import Atom, Module, ConsistencyException
from reactive_modules.parser import parse_atom, parse_module
from reactive_modules.tokenizer import Tokenizer


class TestCase(unittest.TestCase):
    def test_atom(self):
        src = """
        atom Name controls b, x reads a, b 
             awaits  d   ,    c
        initupdate
          [] d' & d' & d' & d' & d' & d' & d' & d' & d' & d' & d' -> b' := a + b; x' := c'
          [] true -> b' := a + b
          [] true ->
          [] d' & d' & d' & d' & d' & d' & d' & d' & d' -> b' := a + b; x' := c'
          [] true ->
        """

        atom: Atom = parse_atom(Tokenizer(src))

        print(atom.pretty_print(max_target_length=40))

        # self.assertEqual(Token('Name', 2, 13, 17), atom.name_token)
        # self.assertEqual([Token('b', 2, 27, 28)], atom.controls_tokens)
        # self.assertEqual([Token('a', 2, 35, 36), Token('b', 2, 38, 39)], atom.reads_tokens)
        # self.assertEqual([Token('c', 3, 30, 31), Token('d', 3, 21, 22)], atom.awaits_tokens)
        #
        # self.assertEqual('atom Name controls b reads a, b awaits c, d', atom.pretty_print())
        # self.assertEqual('atom Name\n'
        #                  'controls b\n'
        #                  'reads a, b\n'
        #                  'awaits c,\n'
        #                  '       d', atom.pretty_print(max_target_length=10))
        # self.assertEqual('atom Name\n'
        #                  'controls\n'
        #                  'b\n'
        #                  'reads\n'
        #                  'a, b\n'
        #                  'awaits\n'
        #                  'c, d', atom.pretty_print(max_target_length=5))

    def test_module(self):
        src = """
        executable Module1 is
        private p1: R
        interface i1, i2: B; i3: N
        external e2, e1, e3, e5: { 1, 2, 3, 4, 5, 6, 7, 8, 9, a, s, d, f, g, e }; e4 : R; b1 : B
        atom A1 controls p1, i1 reads e4, p1
        awaits     i2, b1 
        init
          [] b1' & true -> p1' := 0.1; i1' := b1'
          [] true -> p1' := 0.1; i1' := True
        update
          [] e4 == 1.0 | e4 == 1.0 -> p1' := e4 + p1
          [] i2' & i2' & i2' & i2' & i2' & i2' & i2' & i2' & i2' -> p1' := e4 + p1; i1' := i2'
          [] true ->
        atom A2 controls i2, i3
        initupdate
          [] true -> i2' := False; i3' := 2
        """
        try:
            tokenizer = Tokenizer(src)
            module: Module = parse_module(tokenizer)
            print(module.pretty_print(max_target_length=40))
            self.assertFalse(tokenizer.has_more_token())
        except ConsistencyException as e:
            for c in e.all_causes():
                print(c)
            raise


if __name__ == '__main__':
    unittest.main()
