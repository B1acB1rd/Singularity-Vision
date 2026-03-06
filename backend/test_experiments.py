"""
Phase B: Training Reproducibility Tests

Tests:
1. Config hashing (deterministic)
2. Experiment binding creation
3. Version-safe experiment lookup
4. Reproducibility checks
5. Experiment comparison
"""

import os
import sys
import json
import time
import tempfile
import shutil

backend_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_path)

results = {
    "config_hashing": {"works": [], "slow": [], "breaks": []},
    "experiment_binding": {"works": [], "slow": [], "breaks": []},
    "reproducibility": {"works": [], "slow": [], "breaks": []}
}


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_config_hashing():
    """Test deterministic config hashing."""
    print_section("TEST 1: Config Hashing")
    
    try:
        from core.experiment_tracker import ExperimentConfig
        
        # Test 1a: Create config
        config = ExperimentConfig(
            model_name="yolov8n",
            epochs=50,
            batch_size=16,
            learning_rate=0.01
        )
        
        results["config_hashing"]["works"].append("config_created")
        print(f"✅ ExperimentConfig created")
        
        # Test 1b: Compute hash
        hash1 = config.compute_hash()
        results["config_hashing"]["works"].append(f"hash_computed: {hash1}")
        print(f"✅ Hash computed: {hash1}")
        
        # Test 1c: Determinism - same config = same hash
        config2 = ExperimentConfig(
            model_name="yolov8n",
            epochs=50,
            batch_size=16,
            learning_rate=0.01
        )
        hash2 = config2.compute_hash()
        
        if hash1 == hash2:
            results["config_hashing"]["works"].append("determinism_verified")
            print(f"✅ Determinism verified (identical configs = identical hash)")
        else:
            results["config_hashing"]["breaks"].append("hashes differ for same config")
            print(f"❌ Hashes differ: {hash1} != {hash2}")
        
        # Test 1d: Different config = different hash
        config3 = ExperimentConfig(
            model_name="yolov8n",
            epochs=100,  # Different!
            batch_size=16,
            learning_rate=0.01
        )
        hash3 = config3.compute_hash()
        
        if hash1 != hash3:
            results["config_hashing"]["works"].append("different_config_different_hash")
            print(f"✅ Different epochs → different hash")
        else:
            results["config_hashing"]["breaks"].append("same hash for different config")
        
        # Test 1e: Minor change = different hash
        config4 = ExperimentConfig(
            model_name="yolov8n",
            epochs=50,
            batch_size=16,
            learning_rate=0.001  # Slightly different
        )
        hash4 = config4.compute_hash()
        
        if hash1 != hash4:
            results["config_hashing"]["works"].append("lr_change_detected")
            print(f"✅ LR 0.01 vs 0.001 → different hash")
        else:
            results["config_hashing"]["breaks"].append("same hash for different LR")
        
    except Exception as e:
        results["config_hashing"]["breaks"].append(f"Exception: {str(e)}")
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()


def test_experiment_binding():
    """Test experiment creation with version binding."""
    print_section("TEST 2: Experiment Binding")
    
    temp_dir = tempfile.mkdtemp(prefix="sv_exp_test_")
    
    try:
        from core.experiment_tracker import ExperimentTracker, ExperimentConfig
        
        # Create mock dataset directory
        dataset_dir = os.path.join(temp_dir, "datasets")
        os.makedirs(dataset_dir)
        
        # Create test files
        with open(os.path.join(dataset_dir, "image_001.jpg"), 'wb') as f:
            f.write(b'\xFF\xD8\xFF\xE0' + b'\x00' * 100)
        
        # Test 2a: Create tracker
        tracker = ExperimentTracker(temp_dir)
        results["experiment_binding"]["works"].append("tracker_created")
        print(f"✅ ExperimentTracker created")
        
        # Test 2b: Create experiment
        config = ExperimentConfig(
            model_name="yolov8n",
            epochs=50,
            batch_size=16
        )
        
        binding = tracker.create_experiment(config, dataset_dir)
        
        results["experiment_binding"]["works"].append(
            f"experiment_created: {binding.experiment_id}"
        )
        print(f"✅ Experiment created: {binding.experiment_id}")
        print(f"   Dataset version: {binding.dataset_version[:12]}...")
        print(f"   Config hash: {binding.config_hash}")
        
        # Test 2c: Binding has all required fields
        if all([
            binding.experiment_id,
            binding.dataset_version,
            binding.config_hash,
            binding.config,
            binding.created_at
        ]):
            results["experiment_binding"]["works"].append("binding_complete")
            print(f"✅ Binding has all required fields")
        else:
            results["experiment_binding"]["breaks"].append("binding incomplete")
        
        # Test 2d: Experiment retrievable
        exp = tracker.get_experiment(binding.experiment_id)
        
        if exp and exp["binding"]["experiment_id"] == binding.experiment_id:
            results["experiment_binding"]["works"].append("experiment_retrievable")
            print(f"✅ Experiment retrievable from storage")
        else:
            results["experiment_binding"]["breaks"].append("experiment not retrievable")
        
        # Test 2e: List experiments
        experiments = tracker.list_experiments()
        
        if len(experiments) >= 1:
            results["experiment_binding"]["works"].append(
                f"list_experiments: {len(experiments)}"
            )
            print(f"✅ Listed {len(experiments)} experiment(s)")
        else:
            results["experiment_binding"]["breaks"].append("no experiments listed")
        
        # Test 2f: Update result
        tracker.update_result(
            experiment_id=binding.experiment_id,
            status="running",
            metrics={"epoch": 1, "loss": 0.5, "accuracy": 0.8}
        )
        
        exp_updated = tracker.get_experiment(binding.experiment_id)
        if exp_updated["result"]["status"] == "running":
            results["experiment_binding"]["works"].append("result_updated")
            print(f"✅ Result updated: status=running")
        else:
            results["experiment_binding"]["breaks"].append("result not updated")
        
        # Test 2g: Find matching experiments
        matches = tracker.find_matching_experiments(
            config_hash=binding.config_hash
        )
        
        if binding.experiment_id in matches:
            results["experiment_binding"]["works"].append("find_matching_works")
            print(f"✅ find_matching_experiments: {len(matches)} match(es)")
        else:
            results["experiment_binding"]["breaks"].append("matching not working")
        
    except Exception as e:
        results["experiment_binding"]["breaks"].append(f"Exception: {str(e)}")
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_reproducibility():
    """Test reproducibility checks and comparison."""
    print_section("TEST 3: Reproducibility Checks")
    
    temp_dir = tempfile.mkdtemp(prefix="sv_repro_test_")
    
    try:
        from core.experiment_tracker import ExperimentTracker, ExperimentConfig
        
        # Setup
        dataset_dir = os.path.join(temp_dir, "datasets")
        os.makedirs(dataset_dir)
        with open(os.path.join(dataset_dir, "image.jpg"), 'wb') as f:
            f.write(b'\xFF\xD8\xFF\xE0' + b'\x00' * 100)
        
        tracker = ExperimentTracker(temp_dir)
        
        # Create two experiments with SAME config
        config = ExperimentConfig(
            model_name="yolov8n",
            epochs=50,
            batch_size=16
        )
        
        exp1 = tracker.create_experiment(config, dataset_dir)
        exp2 = tracker.create_experiment(config, dataset_dir)
        
        results["reproducibility"]["works"].append("two_experiments_created")
        print(f"✅ Created 2 experiments with same config")
        
        # Test 3a: Same config = same hash
        if exp1.config_hash == exp2.config_hash:
            results["reproducibility"]["works"].append("same_config_same_hash")
            print(f"✅ Same config → same hash")
        else:
            results["reproducibility"]["breaks"].append("different hash for same config")
        
        # Test 3b: Check reproducibility (should match)
        check = tracker.check_reproducibility(
            experiment_id=exp1.experiment_id,
            dataset_path=dataset_dir,
            config=config
        )
        
        if check["status"] == "match":
            results["reproducibility"]["works"].append("reproducibility_check_match")
            print(f"✅ Reproducibility check: MATCH")
        else:
            results["reproducibility"]["breaks"].append(f"check returned: {check['status']}")
        
        # Test 3c: Check with different config (should mismatch)
        different_config = ExperimentConfig(
            model_name="yolov8s",  # Different model!
            epochs=50,
            batch_size=16
        )
        
        check_mismatch = tracker.check_reproducibility(
            experiment_id=exp1.experiment_id,
            dataset_path=dataset_dir,
            config=different_config
        )
        
        if check_mismatch["status"] == "mismatch":
            results["reproducibility"]["works"].append("mismatch_detected")
            print(f"✅ Config mismatch correctly detected")
        else:
            results["reproducibility"]["breaks"].append("mismatch not detected")
        
        # Test 3d: Compare experiments
        comparison = tracker.compare_experiments([
            exp1.experiment_id,
            exp2.experiment_id
        ])
        
        if comparison.get("config_identical") == True:
            results["reproducibility"]["works"].append("comparison_works")
            print(f"✅ Experiment comparison: configs identical")
        else:
            results["reproducibility"]["breaks"].append("comparison failed")
        
        # Test 3e: Create experiment with different config and compare
        config3 = ExperimentConfig(
            model_name="yolov8n",
            epochs=100,  # Different
            batch_size=32  # Different
        )
        exp3 = tracker.create_experiment(config3, dataset_dir)
        
        comparison2 = tracker.compare_experiments([
            exp1.experiment_id,
            exp3.experiment_id
        ])
        
        if comparison2.get("config_identical") == False:
            results["reproducibility"]["works"].append("different_configs_detected")
            print(f"✅ Different configs correctly detected in comparison")
        else:
            results["reproducibility"]["breaks"].append("different configs not detected")
        
    except Exception as e:
        results["reproducibility"]["breaks"].append(f"Exception: {str(e)}")
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def print_summary():
    """Print test summary."""
    print_section("PHASE B TEST SUMMARY")
    
    for test_name, result in results.items():
        print(f"\n{test_name.upper().replace('_', ' ')}:")
        print(f"  ✅ Works: {len(result['works'])}")
        for item in result['works']:
            print(f"     - {item}")
        
        if result['slow']:
            print(f"  ⚠️ Slow: {len(result['slow'])}")
            for item in result['slow']:
                print(f"     - {item}")
        
        if result['breaks']:
            print(f"  ❌ Breaks: {len(result['breaks'])}")
            for item in result['breaks']:
                print(f"     - {item}")
    
    total_works = sum(len(r['works']) for r in results.values())
    total_slow = sum(len(r['slow']) for r in results.values())
    total_breaks = sum(len(r['breaks']) for r in results.values())
    
    print(f"\n{'='*60}")
    print(f"  OVERALL: ✅ {total_works} | ⚠️ {total_slow} | ❌ {total_breaks}")
    print(f"{'='*60}")
    
    return results


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  PHASE B: TRAINING REPRODUCIBILITY - VALIDATION TESTS")
    print("="*60)
    
    test_config_hashing()
    test_experiment_binding()
    test_reproducibility()
    
    final_results = print_summary()
    
    # Save results
    output_path = os.path.join(backend_path, "experiment_test_results.json")
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
