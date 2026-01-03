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
        
        # Check action format
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
