import unittest

from reactive_modules.tokenizer import Tokenizer


class TokenizerTestCase(unittest.TestCase):
    def test_overlapping_terminals(self):
        tokenizer = Tokenizer("a:b:=c   :  d  :=  e")
        self.assertEqual('a', tokenizer.next_token().token)
        self.assertEqual(':', tokenizer.next_token().token)
        self.assertEqual('b', tokenizer.next_token().token)
        self.assertEqual(':=', tokenizer.next_token().token)
        self.assertEqual('c', tokenizer.next_token().token)
        self.assertEqual(':', tokenizer.next_token().token)
        self.assertEqual('d', tokenizer.next_token().token)
        self.assertEqual(':=', tokenizer.next_token().token)
        self.assertEqual('e', tokenizer.next_token().token)
        self.assertFalse(tokenizer.has_more_token())


if __name__ == '__main__':
    unittest.main()
