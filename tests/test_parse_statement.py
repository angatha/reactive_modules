import unittest

from reactive_modules.cst import TrueFalseExpression, ReferenceExpression
from reactive_modules.parser import parse_boolean_expression
from reactive_modules.tokenizer import Tokenizer


class ParseBooleanExpressionTestCase(unittest.TestCase):
    def test_constant_true(self):
        src = 'truE'
        tokenizer = Tokenizer(src)
        true = parse_boolean_expression(tokenizer, {})
        self.assertFalse(tokenizer.has_more_token())
        self.assertIsInstance(true, TrueFalseExpression)
        self.assertTrue(true.value)

    def test_constant_false(self):
        src = 'FaLse'
        tokenizer = Tokenizer(src)
        false = parse_boolean_expression(tokenizer, {})
        self.assertFalse(tokenizer.has_more_token())
        self.assertIsInstance(false, TrueFalseExpression)
        self.assertFalse(false.value)

    def test_and_constants(self):
        src = 'FaLse & true'
        tokenizer = Tokenizer(src)
        false = parse_boolean_expression(tokenizer, {})
        self.assertFalse(tokenizer.has_more_token())
        self.assertIsInstance(false, TrueFalseExpression)
        self.assertFalse(false.value)

        src = 'True & true'
        tokenizer = Tokenizer(src)
        true = parse_boolean_expression(tokenizer, {})
        self.assertFalse(tokenizer.has_more_token())
        self.assertIsInstance(true, TrueFalseExpression)
        self.assertTrue(true.value)

    def test_and_compression(self):
        src = 'a & false'
        tokenizer = Tokenizer(src)
        false = parse_boolean_expression(tokenizer, {'a'})
        self.assertFalse(tokenizer.has_more_token())
        self.assertIsInstance(false, TrueFalseExpression)
        self.assertFalse(false.value)

        src = 'True ∧ a'
        tokenizer = Tokenizer(src)
        val = parse_boolean_expression(tokenizer, {'a'})
        self.assertFalse(tokenizer.has_more_token())
        self.assertIsInstance(val, ReferenceExpression)
        self.assertEqual('a', val.name)
        self.assertTrue(val.refers_to_old)

    def test_or_constants(self):
        src = 'FaLse ∨ fALSE'
        tokenizer = Tokenizer(src)
        false = parse_boolean_expression(tokenizer, {})
        self.assertFalse(tokenizer.has_more_token())
        self.assertIsInstance(false, TrueFalseExpression)
        self.assertFalse(false.value)

        src = 'True | False'
        tokenizer = Tokenizer(src)
        true = parse_boolean_expression(tokenizer, {})
        self.assertFalse(tokenizer.has_more_token())
        self.assertIsInstance(true, TrueFalseExpression)
        self.assertTrue(true.value)

    def test_or_compression(self):
        src = 'a | true'
        tokenizer = Tokenizer(src)
        true = parse_boolean_expression(tokenizer, {'a'})
        self.assertFalse(tokenizer.has_more_token())
        self.assertIsInstance(true, TrueFalseExpression)
        self.assertTrue(true.value)

        src = "falSE | a'"
        tokenizer = Tokenizer(src)
        val = parse_boolean_expression(tokenizer, {"a'"})
        self.assertFalse(tokenizer.has_more_token())
        self.assertIsInstance(val, ReferenceExpression)
        self.assertEqual('a', val.name)
        self.assertFalse(val.refers_to_old)


if __name__ == '__main__':
    unittest.main()
