#!/usr/bin/env python3
"""
Test script to verify Optuna setup and data availability.
"""

import sys
from pathlib import Path
import joblib
import numpy as np

def test_data_files():
    """Test if all required data files exist."""
    print("🔍 Testing data files...")
    
    # Get the project root directory (two levels up from src/utils)
    project_root = Path(__file__).parent.parent.parent
    pipe_dir = project_root / "production/pipeline"
    required_files = [
        "X_train.joblib",
        "y_train.joblib", 
        "X_test.joblib",
        "y_test.joblib"
    ]
    
    all_exist = True
    for file in required_files:
        file_path = pipe_dir / file
        if file_path.exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING")
            all_exist = False
    
    return all_exist

def test_data_loading():
    """Test if data can be loaded successfully."""
    print("\n📊 Testing data loading...")
    
    try:
        # Get the project root directory
        project_root = Path(__file__).parent.parent.parent
        pipe_dir = project_root / "production/pipeline"
        
        # Load data
        X_train = joblib.load(pipe_dir / "X_train.joblib")
        y_train = joblib.load(pipe_dir / "y_train.joblib")
        X_test = joblib.load(pipe_dir / "X_test.joblib")
        y_test = joblib.load(pipe_dir / "y_test.joblib")
        
        print(f"  ✅ X_train: {X_train.shape}")
        print(f"  ✅ y_train: {y_train.shape}")
        print(f"  ✅ X_test: {X_test.shape}")
        print(f"  ✅ y_test: {y_test.shape}")
        
        # Check class distribution
        print(f"  📈 Train class distribution: {np.bincount(y_train)}")
        print(f"  📈 Test class distribution: {np.bincount(y_test)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error loading data: {e}")
        return False

def test_dependencies():
    """Test if all required dependencies are available."""
    print("\n📦 Testing dependencies...")
    
    dependencies = [
        "optuna",
        "xgboost", 
        "sklearn",
        "numpy",
        "pandas",
        "joblib"
    ]
    
    all_available = True
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"  ✅ {dep}")
        except ImportError:
            print(f"  ❌ {dep} - MISSING")
            all_available = False
    
    return all_available

def test_model_directory():
    """Test if model directory exists and is writable."""
    print("\n📁 Testing model directory...")
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    model_dir = project_root / "production/model"
    
    if model_dir.exists():
        print(f"  ✅ Model directory exists: {model_dir}")
    else:
        print(f"  ⚠️  Model directory doesn't exist, will be created: {model_dir}")
    
    # Test if we can write to the directory
    try:
        test_file = model_dir / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()  # Remove test file
        print(f"  ✅ Model directory is writable")
        return True
    except Exception as e:
        print(f"  ❌ Cannot write to model directory: {e}")
        return False

def test_optuna_storage():
    """Test if Optuna storage is accessible."""
    print("\n🗄️ Testing Optuna storage...")
    
    try:
        import optuna
        # Get the project root directory
        project_root = Path(__file__).parent.parent.parent
        pipe_dir = project_root / "production/pipeline"
        
        # Test creating a study
        storage_url = f"sqlite:///{(pipe_dir / 'test_study.db').as_posix()}"
        study = optuna.create_study(
            study_name="test_study",
            storage=storage_url,
            direction="maximize"
        )
        
        # Clean up
        import os
        os.remove(pipe_dir / "test_study.db")
        
        print(f"  ✅ Optuna storage is working")
        return True
        
    except Exception as e:
        print(f"  ❌ Optuna storage test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 OPTUNA SETUP TEST")
    print("=" * 50)
    
    tests = [
        test_data_files,
        test_data_loading,
        test_dependencies,
        test_model_directory,
        test_optuna_storage
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    test_names = [
        "Data Files",
        "Data Loading", 
        "Dependencies",
        "Model Directory",
        "Optuna Storage"
    ]
    
    all_passed = True
    for name, result in zip(test_names, results):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! You're ready to run Optuna optimization.")
    else:
        print("⚠️  SOME TESTS FAILED. Please fix the issues before running Optuna.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
