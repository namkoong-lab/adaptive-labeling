import unittest
import torch
from k_subset_sampling import SubsetOperator  # Import your SubsetOperator class

class TestSubsetOperator(unittest.TestCase):

    def setUp(self):
        # Common setup for all tests can go here
        self.batch_size = 5
        self.num_samples = 10
        self.k = 2
        self.tau = 0.5
        self.operator_hard = SubsetOperator(k=self.k, tau=self.tau, hard=True)
        self.operator_soft = SubsetOperator(k=self.k, tau=self.tau, hard=False)

    def test_initialization(self):
        self.assertEqual(self.operator_hard.k, self.k)
        self.assertEqual(self.operator_hard.tau, self.tau)
        self.assertTrue(self.operator_hard.hard)
        self.assertFalse(self.operator_soft.hard)

    def test_forward_shape(self):
        scores = torch.randn(self.batch_size, self.num_features)
        output_hard = self.operator_hard(scores)
        output_soft = self.operator_soft(scores)
        self.assertEqual(output_hard.shape, scores.shape)
        self.assertEqual(output_soft.shape, scores.shape)

    def test_hard_output_values(self):
        scores = torch.randn(self.batch_size, self.num_features)
        output = self.operator_hard(scores)
        for row in output:
            self.assertEqual(row.sum().item(), self.k)

    def test_soft_output_values(self):
        scores = torch.randn(self.batch_size, self.num_features)
        output = self.operator_soft(scores)
        for row in output:
            self.assertTrue(abs(row.sum().item() - self.k) < 1e-6)

    def test_error_on_invalid_input(self):
        with self.assertRaises(ValueError):
            invalid_scores = torch.randn(self.batch_size)  # Invalid shape
            self.operator_soft(invalid_scores)

    def test_gradient_flow(self):
        scores = torch.randn(self.batch_size, self.num_features, requires_grad=True)
        output = self.operator_soft(scores)
        self.assertTrue(output.requires_grad)

    def test_different_k_values(self):
        for test_k in [0, 1, self.num_features, self.num_features + 1]:
            operator = SubsetOperator(k=test_k, tau=self.tau, hard=False)
            scores = torch.randn(self.batch_size, self.num_features)
            output = operator(scores)
            # Test the behavior based on the value of test_k
            # ...

    def test_different_tau_values(self):
        for test_tau in [1e-6, 1.0, 100.0]:
            operator = SubsetOperator(k=self.k, tau=test_tau, hard=False)
            scores = torch.randn(self.batch_size, self.num_features)
            output = operator(scores)
            # Test the behavior based on the value of test_tau
            # ...

    def test_batch_size_variability(self):
        for test_batch_size in [1, 5, 20]:
            operator = SubsetOperator(k=self.k, tau=self.tau, hard=False)
            scores = torch.randn(test_batch_size, self.num_features)
            output = operator(scores)
            self.assertEqual(output.shape[0], test_batch_size)

    def test_zero_and_negative_scores(self):
        zero_scores = torch.zeros(self.batch_size, self.num_features)
        negative_scores = -torch.ones(self.batch_size, self.num_features)
        operator = SubsetOperator(k=self.k, tau=self.tau, hard=False)
        # Test with zero_scores and negative_scores
        # ...

    def test_non_2d_input(self):
        non_2d_input = torch.randn(self.batch_size)
        operator = SubsetOperator(k=self.k, tau=self.tau, hard=False)
        with self.assertRaises(ValueError):
            operator(non_2d_input)

    # ... [additional tests]

# ...


if __name__ == '__main__':
    unittest.main()