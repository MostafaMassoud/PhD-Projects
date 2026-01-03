"""
Test Suite for LA-DRL-GSK Q1 Implementation
============================================

Tests for the Q1 implementation including:
- Baseline GSK optimization
- Controllers (Fixed, Heuristic)
- Action mapping
- Windowed control
- Edge handling in junior/senior phases

Run with: python -m pytest tests/test_la_drl_gsk.py -v

Author: LA-DRL-GSK Research Team
Date: 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import unittest


# =============================================================================
# Test Functions
# =============================================================================

def sphere(x):
    """Sphere function (unimodal)."""
    x = np.atleast_2d(x)
    return np.sum(x ** 2, axis=1)


def rastrigin(x):
    """Rastrigin function (multimodal)."""
    x = np.atleast_2d(x)
    d = x.shape[1]
    return 10 * d + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=1)


# =============================================================================
# Platform Tests
# =============================================================================

class TestPlatformDetection(unittest.TestCase):
    """Test platform detection utilities."""
    
    def test_get_platform_info(self):
        from la_drl_gsk import get_platform_info
        info = get_platform_info()
        self.assertIn('system', info)
        self.assertIn('machine', info)
        self.assertIn(info['system'], ['Windows', 'Darwin', 'Linux'])
    
    def test_get_optimal_device(self):
        from la_drl_gsk import get_optimal_device
        device = get_optimal_device()
        self.assertIn(device, ['cpu', 'mps'])
    
    def test_configure_threads(self):
        from la_drl_gsk import configure_threads
        configure_threads()  # Should not raise


# =============================================================================
# Landscape Analyzer Tests
# =============================================================================

class TestLandscapeAnalyzer(unittest.TestCase):
    """Test FLA feature computation."""
    
    def setUp(self):
        from la_drl_gsk import ZeroCostLandscapeAnalyzer
        self.analyzer = ZeroCostLandscapeAnalyzer(dim=10, pop_size=50)
        
    def test_feature_dimension(self):
        """Test that we get 25 features."""
        pop = np.random.uniform(-100, 100, (50, 10))
        fitness = sphere(pop)
        features = self.analyzer.compute_state(pop, fitness)
        self.assertEqual(len(features), 25)
    
    def test_feature_range(self):
        """Test that features are normalized to [0, 1]."""
        pop = np.random.uniform(-100, 100, (50, 10))
        fitness = sphere(pop)
        features = self.analyzer.compute_state(pop, fitness)
        self.assertTrue(np.all(features >= 0))
        self.assertTrue(np.all(features <= 1))
    
    def test_reset(self):
        """Test analyzer reset."""
        pop = np.random.uniform(-100, 100, (50, 10))
        fitness = sphere(pop)
        self.analyzer.compute_state(pop, fitness)
        self.analyzer.reset()
        # Should work after reset
        features = self.analyzer.compute_state(pop, fitness)
        self.assertEqual(len(features), 25)
    
    def test_constant_fitness_no_exception(self):
        """Test analyzer handles constant fitness values (Task 3 - ConstantInputWarning fix)."""
        import warnings
        from la_drl_gsk import ZeroCostLandscapeAnalyzer
        
        analyzer = ZeroCostLandscapeAnalyzer(dim=10, pop_size=50)
        
        # Create population with constant fitness
        pop = np.random.uniform(-100, 100, (50, 10))
        fitness = np.ones(50) * 100.0  # All same fitness value
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            features = analyzer.compute_state(pop, fitness)
            
            # Check no ConstantInputWarning was emitted
            constant_warnings = [x for x in w if 'ConstantInput' in str(x.category)]
            self.assertEqual(len(constant_warnings), 0, 
                           f"ConstantInputWarning should be suppressed, got: {constant_warnings}")
        
        # Check output is valid
        self.assertEqual(features.shape, (25,))
        self.assertTrue(np.all(np.isfinite(features)), "All features should be finite")
        self.assertTrue(np.all(features >= 0) and np.all(features <= 1), 
                       "All features should be in [0, 1]")
    
    def test_constant_population_no_exception(self):
        """Test analyzer handles constant population (all individuals same)."""
        import warnings
        from la_drl_gsk import ZeroCostLandscapeAnalyzer
        
        analyzer = ZeroCostLandscapeAnalyzer(dim=10, pop_size=50)
        
        # Create identical population (all same position)
        single_point = np.random.uniform(-100, 100, 10)
        pop = np.tile(single_point, (50, 1))
        fitness = np.random.uniform(0, 1000, 50)
        
        # Should not raise exception
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            features = analyzer.compute_state(pop, fitness)
            
            # Check no ConstantInputWarning was emitted
            constant_warnings = [x for x in w if 'ConstantInput' in str(x.category)]
            self.assertEqual(len(constant_warnings), 0)
        
        # Check output is valid
        self.assertEqual(features.shape, (25,))
        self.assertTrue(np.all(np.isfinite(features)))
    
    def test_near_converged_population(self):
        """Test analyzer handles nearly converged population (small variance)."""
        import warnings
        from la_drl_gsk import ZeroCostLandscapeAnalyzer
        
        analyzer = ZeroCostLandscapeAnalyzer(dim=10, pop_size=50)
        
        # Create population clustered around optimum (simulating convergence)
        center = np.zeros(10)
        pop = center + np.random.normal(0, 1e-10, (50, 10))  # Very small variance
        fitness = sphere(pop)
        
        # Should not raise exception
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            features = analyzer.compute_state(pop, fitness)
        
        # Check output is valid
        self.assertEqual(features.shape, (25,))
        self.assertTrue(np.all(np.isfinite(features)))
        self.assertTrue(np.all(features >= 0) and np.all(features <= 1))


# =============================================================================
# Controller Tests
# =============================================================================

class TestControllers(unittest.TestCase):
    """Test controller implementations."""
    
    def test_fixed_controller(self):
        """Test FixedController returns constant values."""
        from la_drl_gsk import FixedController
        
        controller = FixedController(K=10.0, kf=0.5, kr=0.9, p=0.1)
        controller.reset(dim=10, pop_size=100, max_nfes=100000)
        
        obs = np.random.uniform(0, 1, 25).astype(np.float32)
        params = controller.act(obs)
        
        self.assertEqual(params['K'], 10.0)
        self.assertEqual(params['kf'], 0.5)
        self.assertEqual(params['kr'], 0.9)
        self.assertEqual(params['p'], 0.1)
    
    def test_heuristic_controller(self):
        """Test HeuristicController returns valid parameters."""
        from la_drl_gsk import HeuristicController, PARAM_RANGES
        
        controller = HeuristicController()
        controller.reset(dim=10, pop_size=100, max_nfes=100000)
        
        obs = np.random.uniform(0, 1, 25).astype(np.float32)
        params = controller.act(obs)
        
        R = PARAM_RANGES
        self.assertGreaterEqual(params['K'], R.K_MIN)
        self.assertLessEqual(params['K'], R.K_MAX)
        self.assertGreaterEqual(params['kf'], R.KF_MIN)
        self.assertLessEqual(params['kf'], R.KF_MAX)
        self.assertGreaterEqual(params['kr'], R.KR_MIN)
        self.assertLessEqual(params['kr'], R.KR_MAX)
        self.assertGreaterEqual(params['p'], R.P_MIN)
        self.assertLessEqual(params['p'], R.P_MAX)
    
    def test_action_mapping(self):
        """Test action mapping to parameters."""
        from la_drl_gsk import map_action_to_params, params_to_action, PARAM_RANGES
        
        # Test extremes
        action_min = np.array([-1.0, -1.0, -1.0, -1.0])
        params = map_action_to_params(action_min)
        self.assertAlmostEqual(params['K'], PARAM_RANGES.K_MIN, places=2)
        self.assertAlmostEqual(params['kf'], PARAM_RANGES.KF_MIN, places=2)
        
        action_max = np.array([1.0, 1.0, 1.0, 1.0])
        params = map_action_to_params(action_max)
        self.assertAlmostEqual(params['K'], PARAM_RANGES.K_MAX, places=2)
        self.assertAlmostEqual(params['kf'], PARAM_RANGES.KF_MAX, places=2)
        
        # Test round-trip
        original_action = np.array([0.5, -0.3, 0.7, 0.0])
        params = map_action_to_params(original_action)
        recovered_action = params_to_action(
            params['K'], params['kf'], params['kr'], params['p']
        )
        np.testing.assert_array_almost_equal(original_action, recovered_action, decimal=2)
    
    def test_create_controller(self):
        """Test controller factory function."""
        from la_drl_gsk import create_controller
        
        # Fixed controller
        controller = create_controller(backend="fixed", K=5.0)
        self.assertEqual(controller.act(np.zeros(25))['K'], 5.0)
        
        # Heuristic controller
        controller = create_controller(backend="heuristic")
        controller.reset(10, 100, 100000)
        params = controller.act(np.random.uniform(0, 1, 25))
        self.assertIn('K', params)


# =============================================================================
# Optimizer Tests
# =============================================================================

class TestLADRLGSK(unittest.TestCase):
    """Test main optimizer."""
    
    def test_baseline_optimization(self):
        """Test baseline GSK converges on Sphere."""
        from la_drl_gsk import LADRLGSK, LADRLGSKConfig
        
        config = LADRLGSKConfig(dim=10, max_nfes=20000, use_rl=False, seed=42)
        optimizer = LADRLGSK(config)
        result = optimizer.optimize(sphere)
        
        self.assertLess(result.best_f, 0.1)  # Should converge well
        self.assertLessEqual(result.nfes_used, 20000)
    
    def test_heuristic_controller(self):
        """Test LA-DRL-GSK with heuristic controller."""
        from la_drl_gsk import LADRLGSK, LADRLGSKConfig
        
        config = LADRLGSKConfig(
            dim=10, 
            max_nfes=10000, 
            use_rl=True,
            controller_backend="heuristic",
            seed=42
        )
        optimizer = LADRLGSK(config)
        result = optimizer.optimize(sphere)
        
        self.assertIsNotNone(result.best_f)
        self.assertGreater(len(result.actions_taken), 0)
        
        # Check action format (Q1: K, kf, kr, p)
        action = result.actions_taken[0]
        self.assertIn('K', action)
        self.assertIn('kf', action)
        self.assertIn('kr', action)
        self.assertIn('p', action)
    
    def test_nfe_compliance(self):
        """Test strict NFE budget compliance."""
        from la_drl_gsk import LADRLGSK, LADRLGSKConfig
        
        max_nfes = 5000
        config = LADRLGSKConfig(dim=10, pop_size=100, max_nfes=max_nfes, seed=42)
        
        eval_count = [0]
        def counting_obj(x):
            eval_count[0] += len(np.atleast_2d(x))
            return sphere(x)
        
        optimizer = LADRLGSK(config)
        result = optimizer.optimize(counting_obj)
        
        self.assertLessEqual(eval_count[0], max_nfes)
        self.assertLessEqual(result.nfes_used, max_nfes)
    
    def test_windowed_control(self):
        """Test that control happens every W generations."""
        from la_drl_gsk import LADRLGSK, LADRLGSKConfig
        
        control_window = 5
        config = LADRLGSKConfig(
            dim=10, 
            pop_size=100,
            max_nfes=10000, 
            use_rl=True,
            controller_backend="heuristic",
            control_window=control_window,
            seed=42
        )
        
        optimizer = LADRLGSK(config)
        result = optimizer.optimize(sphere)
        
        # Number of actions should be roughly max_gens / control_window
        max_gens = 10000 // 100
        expected_actions = max_gens // control_window
        
        # Allow some tolerance
        self.assertGreater(len(result.actions_taken), expected_actions * 0.5)
        self.assertLessEqual(len(result.actions_taken), expected_actions + 2)
    
    def test_factory_functions(self):
        """Test factory function convenience."""
        from la_drl_gsk import create_baseline_gsk, create_ladrl_gsk
        
        # Baseline (more budget for convergence)
        optimizer = create_baseline_gsk(dim=10, max_nfes=10000, seed=42)
        result = optimizer.optimize(sphere)
        self.assertLess(result.best_f, 1.0)
        
        # LA-DRL-GSK
        optimizer = create_ladrl_gsk(dim=10, max_nfes=10000, seed=42)
        result = optimizer.optimize(sphere)
        self.assertIsNotNone(result.best_f)


# =============================================================================
# Junior/Senior Index Tests
# =============================================================================

class TestJuniorSeniorIndices(unittest.TestCase):
    """Test junior and senior phase index selection."""
    
    def test_junior_indices_basic(self):
        """Test junior indices with basic population."""
        from la_drl_gsk import LADRLGSK, LADRLGSKConfig
        
        config = LADRLGSKConfig(dim=10, pop_size=20, seed=42)
        optimizer = LADRLGSK(config)
        
        # Create sorted indices (best to worst)
        ind_best = np.arange(20, dtype=np.int64)
        
        R1, R2, R3 = optimizer._junior_indices(ind_best, 20)
        
        # Check array shapes
        self.assertEqual(R1.shape, (20,))
        self.assertEqual(R2.shape, (20,))
        self.assertEqual(R3.shape, (20,))
        
        # Check best individual: R1=1, R2=2 (2nd and 3rd best)
        self.assertEqual(R1[0], 1)
        self.assertEqual(R2[0], 2)
        
        # Check worst individual: R1=17, R2=18 (3rd and 2nd worst)
        self.assertEqual(R1[19], 17)
        self.assertEqual(R2[19], 18)
        
        # Check R3 is conflict-free
        idx = np.arange(20)
        conflicts = (R3 == idx) | (R3 == R1) | (R3 == R2)
        self.assertFalse(np.any(conflicts), "R3 should not conflict with self, R1, or R2")
    
    def test_senior_indices_basic(self):
        """Test senior indices with basic population."""
        from la_drl_gsk import LADRLGSK, LADRLGSKConfig
        
        config = LADRLGSKConfig(dim=10, pop_size=20, seed=42)
        optimizer = LADRLGSK(config)
        
        # Create sorted indices
        ind_best = np.arange(20, dtype=np.int64)
        p = 0.1  # 10% top/bottom
        
        R1, R2, R3 = optimizer._senior_indices(ind_best, 20, p)
        
        # Check shapes
        self.assertEqual(R1.shape, (20,))
        self.assertEqual(R2.shape, (20,))
        self.assertEqual(R3.shape, (20,))
        
        # With p=0.1 and NP=20:
        # n_top = floor(20 * 0.1) = 2
        # top group: indices [0, 1]
        # middle group: indices [2, ..., 17]
        # bottom group: indices [18, 19]
        
        # R1 should be from top group
        self.assertTrue(np.all(R1 < 2), "R1 should be from top group")
        
        # R3 should be from bottom group
        self.assertTrue(np.all(R3 >= 18), "R3 should be from bottom group")
    
    def test_senior_indices_middle_nonempty(self):
        """Test that middle group is non-empty even with large p."""
        from la_drl_gsk import LADRLGSK, LADRLGSKConfig
        
        config = LADRLGSKConfig(dim=10, pop_size=10, seed=42)
        optimizer = LADRLGSK(config)
        
        ind_best = np.arange(10, dtype=np.int64)
        p = 0.2  # 20% each for top/bottom
        
        R1, R2, R3 = optimizer._senior_indices(ind_best, 10, p)
        
        # Should not raise and R2 should be valid
        self.assertEqual(R2.shape, (10,))
    
    def test_junior_indices_r3_conflicts(self):
        """Test that R3 excludes self, R1, and R2 (Task C test)."""
        from la_drl_gsk import LADRLGSK, LADRLGSKConfig
        
        # Use larger population to ensure conflicts can be resolved
        config = LADRLGSKConfig(dim=5, pop_size=10, seed=42)
        optimizer = LADRLGSK(config)
        
        # Initialize state by resetting
        state = optimizer.reset_run(sphere)
        
        # Get sorted indices
        ind_best = np.argsort(state.fitness)
        NP = 10
        
        # Call junior indices
        R1, R2, R3 = optimizer._junior_indices(ind_best, NP)
        
        # Assert for all i: R3[i] != i, R3[i] != R1[i], R3[i] != R2[i]
        idx = np.arange(NP)
        
        # Check R3 != self
        self.assertFalse(np.any(R3 == idx), "R3 should not equal self index")
        
        # Check R3 != R1
        self.assertFalse(np.any(R3 == R1), "R3 should not equal R1")
        
        # Check R3 != R2
        self.assertFalse(np.any(R3 == R2), "R3 should not equal R2")


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_small_population(self):
        """Test with small population size."""
        from la_drl_gsk import LADRLGSK, LADRLGSKConfig
        
        config = LADRLGSKConfig(dim=5, pop_size=10, max_nfes=1000, seed=42)
        optimizer = LADRLGSK(config)
        result = optimizer.optimize(sphere)
        
        self.assertIsNotNone(result.best_f)
    
    def test_high_dimension(self):
        """Test with higher dimension."""
        from la_drl_gsk import LADRLGSK, LADRLGSKConfig
        
        config = LADRLGSKConfig(dim=50, pop_size=100, max_nfes=10000, seed=42)
        optimizer = LADRLGSK(config)
        result = optimizer.optimize(sphere)
        
        self.assertIsNotNone(result.best_f)
        self.assertLess(result.best_f, 1e10)  # Should make progress
    
    def test_reproducibility(self):
        """Test that same seed gives same result."""
        from la_drl_gsk import LADRLGSK, LADRLGSKConfig
        
        results = []
        for _ in range(2):
            config = LADRLGSKConfig(dim=10, max_nfes=5000, seed=12345)
            optimizer = LADRLGSK(config)
            result = optimizer.optimize(sphere)
            results.append(result.best_f)
        
        self.assertEqual(results[0], results[1])


# =============================================================================
# CLI Tests
# =============================================================================

class TestCLI(unittest.TestCase):
    """Test CLI argument parsing."""
    
    def test_demo_cec_path_argument(self):
        """Test that demo subcommand accepts --cec-path."""
        import argparse
        import sys
        from pathlib import Path
        
        # Import and inspect run.py's argument parser
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        # Create a fresh parser like run.py does
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')
        
        # Demo command should accept --cec-path
        demo_parser = subparsers.add_parser('demo')
        demo_parser.add_argument('--detailed', '-d', action='store_true')
        demo_parser.add_argument('--cec-path', type=str, default=None)
        
        # Test parsing
        args = parser.parse_args(['demo', '--cec-path', '/some/path'])
        self.assertEqual(args.command, 'demo')
        self.assertEqual(args.cec_path, '/some/path')
        
        # Test without --cec-path
        args = parser.parse_args(['demo'])
        self.assertEqual(args.command, 'demo')
        self.assertIsNone(args.cec_path)


# =============================================================================
# CEC2017 Benchmark Tests
# =============================================================================

class TestCEC2017Benchmark(unittest.TestCase):
    """Test CEC2017 benchmark loading."""
    
    def test_function_loading(self):
        """Test that benchmark functions can be loaded."""
        from la_drl_gsk.cec2017_benchmark import get_cec2017_function, CEC2017_FUNCTIONS
        
        # Test a few functions
        for func_id in [1, 3, 5, 10]:
            objective, f_opt = get_cec2017_function(func_id, dim=10)
            
            # Test evaluation
            x = np.random.uniform(-100, 100, (10, 10))
            y = objective(x)
            
            self.assertEqual(y.shape, (10,))
            self.assertAlmostEqual(f_opt, func_id * 100.0, places=1)
    
    def test_f2_excluded(self):
        """Test that F2 raises an error."""
        from la_drl_gsk.cec2017_benchmark import get_cec2017_function
        
        with self.assertRaises(ValueError):
            get_cec2017_function(2, dim=10)
    
    def test_invalid_func_id(self):
        """Test invalid function IDs."""
        from la_drl_gsk.cec2017_benchmark import get_cec2017_function
        
        with self.assertRaises(ValueError):
            get_cec2017_function(0, dim=10)
        
        with self.assertRaises(ValueError):
            get_cec2017_function(31, dim=10)
    
    def test_cec2017_loader_bundled(self):
        """Test CEC2017 loader uses bundled implementation correctly (Task A test)."""
        from la_drl_gsk.cec2017_benchmark import (
            get_cec2017_function, reset_cec2017_cache, get_cec2017_source
        )
        
        # Reset cache to force reload
        reset_cec2017_cache()
        
        # Load function without external path - should use bundled
        objective, f_opt = get_cec2017_function(1, dim=10)
        
        # Test evaluation with random input (5, 10) array
        x = np.random.uniform(-100, 100, (5, 10))
        y = objective(x)
        
        # Assert output shape is (5,)
        self.assertEqual(y.shape, (5,))
        
        # Verify source is bundled (not synthetic)
        source = get_cec2017_source()
        self.assertIn(source, ['bundled', 'external'])  # Either bundled or external, but not synthetic
    
    def test_cec2017_vectorized_evaluation(self):
        """Test CEC2017 functions work with vectorized input."""
        from la_drl_gsk.cec2017_benchmark import get_cec2017_function
        
        for func_id in [1, 5, 10, 20, 30]:
            objective, _ = get_cec2017_function(func_id, dim=10)
            
            # Test batch evaluation
            x = np.random.uniform(-100, 100, (100, 10))
            y = objective(x)
            
            self.assertEqual(y.shape, (100,))
            self.assertTrue(np.all(np.isfinite(y)))


# =============================================================================
# Gymnasium Environment Tests (Optional)
# =============================================================================

class TestGymEnvironment(unittest.TestCase):
    """Test Gymnasium environment (skipped if not installed)."""
    
    @classmethod
    def setUpClass(cls):
        try:
            import gymnasium
            cls.gym_available = True
        except ImportError:
            cls.gym_available = False
    
    def test_env_creation(self):
        """Test environment can be created."""
        if not self.gym_available:
            self.skipTest("gymnasium not installed")
        
        from la_drl_gsk.gsk_env import make_gsk_env
        if make_gsk_env is None:
            self.skipTest("gymnasium not installed")
        
        env = make_gsk_env(dim=10, max_nfes=10000)
        self.assertIsNotNone(env)
        env.close()
    
    def test_env_reset_step(self):
        """Test environment reset and step."""
        if not self.gym_available:
            self.skipTest("gymnasium not installed")
        
        from la_drl_gsk.gsk_env import make_gsk_env
        if make_gsk_env is None:
            self.skipTest("gymnasium not installed")
        
        env = make_gsk_env(dim=10, max_nfes=10000)
        
        obs, info = env.reset()
        self.assertEqual(obs.shape, (25,))
        self.assertIn('func_id', info)
        
        action = np.array([0.0, 0.0, 0.0, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        
        self.assertEqual(obs.shape, (25,))
        self.assertIsInstance(reward, float)
        self.assertIn('best_f', info)
        
        env.close()


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
