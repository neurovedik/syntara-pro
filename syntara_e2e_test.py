#!/usr/bin/env python3
"""
=============================================================================
SYNTARA-PRO: End-to-End Test Suite
=============================================================================

Comprehensive testing of all 32+ modules with pass/fail reporting.
Tests individual modules, integration, and full system workflows.

Run: python syntara_e2e_test.py
=============================================================================
"""

import sys
import time
import traceback
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# Test Results Container
@dataclass
class TestResult:
    name: str
    passed: bool
    duration: float
    error: str = None
    details: Dict = None

class SyntaraE2ETestSuite:
    """Complete E2E Test Suite for SYNTARA-PRO."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.modules_tested = 0
        self.modules_passed = 0
        
    # ========================================================================
    # TEST UTILITIES
    # ========================================================================
    
    def run_test(self, test_name: str, test_func) -> TestResult:
        """Execute a single test with timing and error handling."""
        start = time.time()
        try:
            result = test_func()
            duration = time.time() - start
            return TestResult(
                name=test_name,
                passed=True,
                duration=duration,
                details=result if isinstance(result, dict) else None
            )
        except Exception as e:
            duration = time.time() - start
            return TestResult(
                name=test_name,
                passed=False,
                duration=duration,
                error=str(e),
                details={'traceback': traceback.format_exc()}
            )
    
    def assert_true(self, condition: bool, message: str = "Assertion failed"):
        """Assertion helper."""
        if not condition:
            raise AssertionError(message)
    
    def assert_almost_equal(self, a: float, b: float, tolerance: float = 0.01):
        """Float comparison."""
        if abs(a - b) > tolerance:
            raise AssertionError(f"{a} != {b} (tolerance: {tolerance})")
    
    # ========================================================================
    # BASE MODULE TESTS
    # ========================================================================
    
    def test_neural_params(self) -> Dict:
        """Test 1: Neuron Parameters."""
        from syntara_core import NeuronParams, AdExNeuron
        
        params = NeuronParams(tau_m=20.0, V_th=-50.0)
        neuron = AdExNeuron(params, neuron_id=0)
        
        # Test single step
        spiked = neuron.step(1.0, dt=0.1)
        
        self.assert_true(hasattr(neuron, 'V'), "Neuron should have voltage")
        self.assert_true(spiked is False or spiked is True, "Spike should be boolean")
        
        return {'neuron_id': neuron.id, 'voltage': neuron.V}
    
    def test_spiking_network(self) -> Dict:
        """Test 2: Liquid Spiking Network."""
        from syntara_core import LiquidSpikingNetwork, SyntaraConfig
        
        config = SyntaraConfig()
        lsn = LiquidSpikingNetwork(
            config=config,
            n_excitatory=100,
            n_inhibitory=25,
            connectivity=0.2,
            input_dim=50
        )
        
        # Stimulate
        input_vec = __import__('numpy').random.randn(50)
        currents = lsn.stimulate(input_vec)
        
        # Run steps
        for _ in range(10):
            spikes = lsn.run_step()
        
        stats = lsn.get_stats()
        self.assert_true(stats['n_neurons'] == 125, "Should have 125 neurons")
        self.assert_true(stats['total_spikes'] >= 0, "Spikes should be non-negative")
        
        return stats
    
    def test_hypervector(self) -> Dict:
        """Test 3: Hyperdimensional Computing."""
        from syntara_core import HyperVectorEngine, SyntaraConfig
        
        config = SyntaraConfig(hv_dim=1000)
        hve = HyperVectorEngine(config)
        
        # Generate and test
        hv1 = hve.generate_random()
        hv2 = hve.generate_random()
        
        bound = hve.bind(hv1, hv2)
        bundled = hve.bundle([hv1, hv2])
        sim = hve.similarity(hv1, hv2)
        
        self.assert_true(len(hv1) == 1000, "Hypervector should be 1000D")
        self.assert_true(-1 <= sim <= 1, "Similarity should be in [-1, 1]")
        self.assert_true(len(bound) == 1000, "Bound vector should maintain dimension")
        
        # Test encoding
        text_hv = hve.encode_text("test intelligence")
        self.assert_true(len(text_hv) == 1000, "Text encoding should work")
        
        return {'similarity': sim, 'vocab_size': len(hve.vocab)}
    
    def test_causal_reasoning(self) -> Dict:
        """Test 4: Causal Reasoning."""
        from syntara_core import CausalGraph, StructuralEquation
        
        cg = CausalGraph()
        cg.add_variable('A', parents=[])
        cg.add_variable('B', parents=['A'])
        cg.add_variable('C', parents=['A', 'B'])
        
        cg.add_equation('A', StructuralEquation(lambda: 1.0))
        cg.add_equation('B', StructuralEquation(lambda A: 0.5 * A))
        cg.add_equation('C', StructuralEquation(lambda A, B: A + B))
        
        # Compute
        values = cg._compute({})
        self.assert_true('A' in values, "A should be computed")
        self.assert_true('C' in values, "C should be computed")
        
        # Intervention
        intervened = cg.intervene('B', 2.0)
        self.assert_true(intervened['B'] == 2.0, "Intervention should work")
        
        return values
    
    def test_holographic_memory(self) -> Dict:
        """Test 5: Holographic Memory."""
        from syntara_core import HolographicMemory
        import numpy as np
        
        memory = HolographicMemory(capacity=100, dim=512)
        
        # Store
        key1 = np.random.randn(512)
        memory.store(key1, "value1", associations=["test"])
        
        key2 = np.random.randn(512)
        memory.store(key2, "value2")
        
        # Retrieve
        query = key1 + np.random.randn(512) * 0.1
        results = memory.retrieve(query, top_k=3)
        
        self.assert_true(len(results) > 0, "Should retrieve results")
        self.assert_true(len(memory.traces) == 2, "Should have 2 traces")
        
        stats = memory.get_stats()
        return stats
    
    def test_meta_compiler(self) -> Dict:
        """Test 6: Meta-Compiler."""
        from syntara_core import MetaCompiler
        
        compiler = MetaCompiler(optimization_level=2)
        
        # Analyze code
        code = "def test():\n    return 42"
        analysis = compiler.analyze_code(code)
        
        self.assert_true(analysis['n_functions'] == 1, "Should detect 1 function")
        
        # Generate function
        neural_func = compiler.generate_function('neural_layer', {
            'layer_name': 'test_layer'
        })
        
        self.assert_true(callable(neural_func), "Should generate callable")
        
        stats = compiler.get_stats()
        return stats
    
    def test_cellular_automata(self) -> Dict:
        """Test 7: Cellular Automata."""
        from syntara_core import CellularAutomata
        
        ca = CellularAutomata(grid_size=(10, 10), state_dim=8)
        
        # Evolve
        initial_state = ca.get_grid_state()
        ca.evolve(steps=5)
        final_state = ca.get_grid_state()
        
        self.assert_true(len(initial_state) == len(final_state), "State size should be constant")
        self.assert_true(ca.grid_size == (10, 10), "Grid size should be maintained")
        
        stats = ca.get_stats()
        return stats
    
    # ========================================================================
    # PERCEPTION TESTS
    # ========================================================================
    
    def test_nlp_engine(self) -> Dict:
        """Test 8: NLP Engine."""
        from syntara_core import NLPEngine
        
        nlp = NLPEngine(vocab_size=256, embedding_dim=128)
        
        text = "Artificial intelligence is powerful"
        result = nlp.process(text)
        
        self.assert_true(result['n_tokens'] > 0, "Should tokenize")
        self.assert_true(len(result['keywords']) > 0, "Should extract keywords")
        self.assert_true(result['embedding_shape'][1] == 128, "Embedding dim should match")
        
        # Compare
        sim = nlp.compare("hello world", "hello there")
        self.assert_true('similarity' in sim, "Should compute similarity")
        
        return {'tokens': result['n_tokens'], 'keywords': result['keywords'][:3]}
    
    def test_vision_engine(self) -> Dict:
        """Test 9: Computer Vision."""
        from syntara_core import ComputerVisionEngine, Image
        import numpy as np
        
        cv = ComputerVisionEngine(input_shape=(224, 224, 3))
        
        # Create test image
        img = Image(data=np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
        
        # Test classification
        result = cv.classify_image(img)
        self.assert_true('top_prediction' in result, "Should classify")
        self.assert_true('confidence' in result, "Should have confidence")
        
        # Test edge detection
        edges = cv.detect_edges(img, method='sobel')
        self.assert_true(edges.shape == (224, 224), "Edge map should be 2D")
        
        # Test feature extraction
        features = cv.extract_features(img, method='cnn')
        self.assert_true(len(features) > 0, "Should extract features")
        
        return {'classification': result['top_prediction']['class']}
    
    # ========================================================================
    # ADVANCED MODULE TESTS
    # ========================================================================
    
    def test_quantum_computing(self) -> Dict:
        """Test 10: Quantum Computing."""
        from syntara_core import QuantumComputingEngine
        
        quantum = QuantumComputingEngine(n_qubits=6, n_gates=10)
        
        # Apply gates
        quantum.apply_gate(0, target_qubit=0)
        quantum.apply_gate(1, target_qubit=1)
        
        # Measure
        measurements = quantum.measure(n_samples=100)
        self.assert_true(len(measurements) > 0, "Should measure states")
        
        # Quantum search
        prob = quantum.quantum_search(target=5, iterations=3)
        self.assert_true(0 <= prob <= 1, "Probability should be valid")
        
        stats = quantum.get_stats()
        return stats
    
    def test_evolution(self) -> Dict:
        """Test 11: Neuromorphic Evolution."""
        from syntara_core import NeuromorphicEvolution
        import numpy as np
        
        evolution = NeuromorphicEvolution(
            population_size=10,
            mutation_rate=0.02
        )
        
        evolution.initialize_population([10, 20, 5])
        
        # Evolve a few generations
        X = np.random.randn(50, 10)
        y = np.random.randn(50, 5)
        
        for _ in range(3):
            evolution.evolve_generation((X, y))
        
        best = evolution.get_best()
        self.assert_true(best is not None, "Should have best individual")
        self.assert_true(evolution.generation == 3, "Should have 3 generations")
        
        stats = evolution.get_stats()
        return stats
    
    def test_consciousness(self) -> Dict:
        """Test 12: Global Workspace Consciousness."""
        from syntara_core import GlobalWorkspaceTheory
        
        gw = GlobalWorkspaceTheory(workspace_capacity=5, broadcast_threshold=0.6)
        
        # Perceive
        gw.perceive("visual_input", modality='visual', intensity=0.8)
        gw.perceive("audio_input", modality='auditory', intensity=0.7)
        
        contents = gw.get_conscious_contents()
        self.assert_true(len(contents) > 0, "Should have conscious contents")
        
        # Form intention
        intention = gw.form_intention("complete_task")
        self.assert_true(intention['formed'], "Should form intention")
        
        # Self-reflect
        reflection = gw.self_reflect()
        self.assert_true('conscious_contents' in reflection, "Should self-reflect")
        
        return reflection
    
    def test_world_model(self) -> Dict:
        """Test 13: Predictive World Model."""
        from syntara_core import PredictiveWorldModel
        import numpy as np
        
        model = PredictiveWorldModel(state_dim=32, action_dim=8)
        
        # Learn from experience
        for _ in range(20):
            state = np.random.randn(32)
            action = np.random.randn(8)
            next_state = state + 0.1 * np.random.randn(32)
            reward = np.random.randn()
            model.learn_from_experience(state, action, next_state, reward)
        
        # Predict
        state = np.random.randn(32)
        action = np.random.randn(8)
        pred = model.predict_next_state(state, action)
        
        self.assert_true(len(pred) == 32, "Prediction should match state dim")
        
        stats = model.get_stats()
        return stats
    
    def test_creativity(self) -> Dict:
        """Test 14: Emergent Creativity."""
        from syntara_core import EmergentCreativityEngine
        
        creativity = EmergentCreativityEngine(concept_space_dim=128, n_concepts=10)
        
        # Add concepts
        creativity.add_concept('technology')
        creativity.add_concept('art')
        
        # Blend
        blend = creativity.conceptual_blend('technology', 'art', 'tech_art')
        self.assert_true('new_concept' in blend, "Should create blend")
        
        # Generate idea
        idea = creativity.generate_novel_idea(domain='innovation')
        self.assert_true('novelty_score' in idea, "Should have novelty score")
        self.assert_true(0 <= idea['novelty_score'] <= 1, "Novelty should be in [0,1]")
        
        return {'n_concepts': len(creativity.concepts)}
    
    def test_swarm(self) -> Dict:
        """Test 15: Swarm Intelligence."""
        from syntara_core import SwarmIntelligenceNetwork
        
        swarm = SwarmIntelligenceNetwork(n_agents=20, behavior='flocking')
        
        # Simulate
        metrics = swarm.simulate(n_steps=20)
        
        self.assert_true(metrics['n_agents'] == 20, "Should have 20 agents")
        self.assert_true('cohesion' in metrics, "Should compute cohesion")
        
        # Decision
        decision, conf = swarm.collective_decision(['A', 'B', 'C'])
        self.assert_true(decision in ['A', 'B', 'C'], "Should make valid decision")
        
        return metrics
    
    def test_temporal(self) -> Dict:
        """Test 16: Temporal Reasoning."""
        from syntara_core import TemporalReasoningEngine
        
        temporal = TemporalReasoningEngine()
        
        # Record events
        temporal.record_event('start', {'value': 1})
        temporal.record_event('process', {'value': 2})
        temporal.record_event('success', {'value': 3})
        
        self.assert_true(len(temporal.event_history) == 3, "Should have 3 events")
        
        # Predict
        predictions = temporal.predict_future(n_predictions=2)
        
        # Causal chain
        if temporal.event_history:
            chain = temporal.find_causal_chain(temporal.event_history[-1].event_id)
        
        stats = temporal.get_stats()
        return stats
    
    def test_self_replication(self) -> Dict:
        """Test 17: Self-Replication."""
        from syntara_core import SelfReplicator
        
        replicator = SelfReplicator(mutation_rate=0.1)
        
        # Analyze code
        code = "def test(x):\n    return x * 2"
        analysis = replicator.analyze_code(code)
        
        self.assert_true(analysis['n_functions'] == 1, "Should detect function")
        
        # Create variant
        variant = replicator.create_variant(code, 'vectorization')
        self.assert_true(len(variant) > 0, "Should create variant")
        
        # Evaluate
        score = replicator.evaluate_variant(variant, [{}])
        self.assert_true(0 <= score <= 1, "Score should be valid")
        
        return {'score': score}
    
    # ========================================================================
    # AGENTIC & ACTION TESTS
    # ========================================================================
    
    def test_agentic_executor(self) -> Dict:
        """Test 18: Agentic Task Executor."""
        from syntara_core import AgenticExecutor
        
        agent = AgenticExecutor(max_depth=3, retry_limit=2)
        
        # Register custom tool
        def custom_tool(x: int) -> Dict:
            return {'success': True, 'result': x * 2}
        
        agent.register_tool('doubler', custom_tool)
        
        # Execute
        result = agent.run("calculate sqrt(16)")
        self.assert_true('execution' in result, "Should have execution result")
        
        return {'tools': agent.get_stats()['tools_registered']}
    
    # ========================================================================
    # WEB & EXTERNAL TESTS
    # ========================================================================
    
    def test_web_search(self) -> Dict:
        """Test 19: Web Search."""
        from syntara_core import WebSearch
        
        web = WebSearch(max_results=5)
        
        results = web.search("neural networks", n_results=3)
        self.assert_true(len(results) == 3, "Should return 3 results")
        self.assert_true('title' in results[0], "Results should have titles")
        
        content = web.extract_content("https://example.com")
        self.assert_true('success' in content, "Should extract content")
        
        return {'n_results': len(results)}
    
    # ========================================================================
    # INTEGRATION TESTS
    # ========================================================================
    
    def test_full_system_init(self) -> Dict:
        """Test 20: Full SYNTARA-PRO Initialization."""
        from syntara_core import SyntaraPRO, SyntaraUltimateConfig
        
        config = SyntaraUltimateConfig(
            enable_spiking=True,
            enable_hypervector=True,
            enable_causal=True,
            enable_memory=True,
            enable_nlp=True,
            enable_agentic=True,
            enable_quantum=True,
            enable_evolution=True,
            enable_consciousness=True,
            enable_world_model=True,
            enable_creativity=True,
            enable_swarm=True,
            enable_temporal=True,
            enable_self_replication=True,
            enable_vision=True,
            enable_rl=True,
            enable_federated=True,
            enable_knowledge_graph=True,
            enable_predictive=True,
            agi_level=5
        )
        
        syntara = SyntaraPRO(config)
        
        self.assert_true(syntara.initialized, "Should initialize")
        self.assert_true(len(syntara.modules) > 20, "Should have 20+ modules")
        
        stats = syntara.get_stats()
        return stats
    
    def test_universal_processing(self) -> Dict:
        """Test 21: Universal Processing Interface."""
        from syntara_core import SyntaraPRO, SyntaraUltimateConfig
        import numpy as np
        
        config = SyntaraUltimateConfig(enable_nlp=True, enable_spiking=True)
        syntara = SyntaraPRO(config)
        
        # Test text processing
        result = syntara.process("Test input", task_type='text')
        self.assert_true(result.get('success'), "Text processing should succeed")
        
        # Test neural processing
        result2 = syntara.process(np.random.randn(100), task_type='neural')
        self.assert_true(result2.get('success'), "Neural processing should succeed")
        
        return {'modules_used_1': result.get('modules_used'),
                'modules_used_2': result2.get('modules_used')}
    
    def test_deep_thinking(self) -> Dict:
        """Test 22: Deep Thinking Mode."""
        from syntara_core import SyntaraPRO, SyntaraUltimateConfig
        
        config = SyntaraUltimateConfig(enable_fractal=True, enable_creativity=True)
        syntara = SyntaraPRO(config)
        
        result = syntara.think("Test problem", depth=2)
        
        self.assert_true('reasoning_chain' in result, "Should have reasoning chain")
        self.assert_true('problem' in result, "Should track problem")
        
        return {'chain_length': len(result.get('reasoning_chain', []))}
    
    def test_meta_learning(self) -> Dict:
        """Test 23: Meta-Learning."""
        from syntara_core import SyntaraPRO, SyntaraUltimateConfig
        import numpy as np
        
        config = SyntaraUltimateConfig(enable_memory=True, enable_temporal=True)
        syntara = SyntaraPRO(config)
        
        experience = {
            'type': 'test',
            'state': np.random.randn(64),
            'action': np.random.randn(10),
            'reward': 0.5
        }
        
        result = syntara.learn(experience)
        self.assert_true(result.get('learned'), "Should learn")
        
        return result
    
    def test_self_reflection(self) -> Dict:
        """Test 24: Self-Reflection."""
        from syntara_core import SyntaraPRO, SyntaraUltimateConfig
        
        config = SyntaraUltimateConfig()
        syntara = SyntaraPRO(config)
        
        # Add some execution history
        syntara.execution_history = [
            {'latency': 0.1, 'success': True},
            {'latency': 0.2, 'success': True}
        ]
        
        reflection = syntara.self_reflect()
        
        self.assert_true('n_modules_active' in reflection, "Should report modules")
        self.assert_true('success_rate' in reflection, "Should report success rate")
        
        return reflection
    
    # ========================================================================
    # RUN ALL TESTS
    # ========================================================================
    
    def run_all_tests(self):
        """Execute complete E2E test suite."""
        print("\n" + "="*70)
        print("🧪 SYNTARA-PRO END-TO-END TEST SUITE")
        print("="*70)
        print(f"Total tests: 24+")
        print(f"Modules to test: 32+")
        print()
        
        tests = [
            # Base modules
            ("Neuron Parameters", self.test_neural_params),
            ("Spiking Network", self.test_spiking_network),
            ("Hypervector Computing", self.test_hypervector),
            ("Causal Reasoning", self.test_causal_reasoning),
            ("Holographic Memory", self.test_holographic_memory),
            ("Meta-Compiler", self.test_meta_compiler),
            ("Cellular Automata", self.test_cellular_automata),
            
            # Perception
            ("NLP Engine", self.test_nlp_engine),
            ("Vision Engine", self.test_vision_engine),
            
            # Advanced
            ("Quantum Computing", self.test_quantum_computing),
            ("Evolution", self.test_evolution),
            ("Consciousness", self.test_consciousness),
            ("World Model", self.test_world_model),
            ("Creativity", self.test_creativity),
            ("Swarm Intelligence", self.test_swarm),
            ("Temporal Reasoning", self.test_temporal),
            ("Self-Replication", self.test_self_replication),
            
            # Action
            ("Agentic Executor", self.test_agentic_executor),
            
            # External
            ("Web Search", self.test_web_search),
            
            # Integration
            ("Full System Init", self.test_full_system_init),
            ("Universal Processing", self.test_universal_processing),
            ("Deep Thinking", self.test_deep_thinking),
            ("Meta Learning", self.test_meta_learning),
            ("Self Reflection", self.test_self_reflection),
        ]
        
        passed = 0
        failed = 0
        total_time = 0
        
        for i, (name, test_func) in enumerate(tests, 1):
            result = self.run_test(name, test_func)
            self.results.append(result)
            total_time += result.duration
            
            status = "✅ PASS" if result.passed else "❌ FAIL"
            if result.passed:
                passed += 1
            else:
                failed += 1
            
            print(f"{i:2d}. {status} | {name:30s} | {result.duration:.3f}s")
            
            if not result.passed and result.error:
                print(f"     Error: {result.error[:50]}...")
        
        # Summary
        print("\n" + "="*70)
        print("📊 TEST SUMMARY")
        print("="*70)
        print(f"Total tests: {len(tests)}")
        print(f"Passed: {passed} ✅")
        print(f"Failed: {failed} ❌")
        print(f"Success rate: {passed/len(tests)*100:.1f}%")
        print(f"Total time: {total_time:.3f}s")
        print(f"Avg time/test: {total_time/len(tests):.3f}s")
        print("="*70)
        
        if failed == 0:
            print("🎉 ALL TESTS PASSED! SYNTARA-PRO IS FULLY OPERATIONAL!")
        else:
            print(f"⚠️ {failed} test(s) failed. Review errors above.")
        
        return passed, failed


# ========================================================================
# MAIN
# ========================================================================

if __name__ == "__main__":
    suite = SyntaraE2ETestSuite()
    passed, failed = suite.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)
