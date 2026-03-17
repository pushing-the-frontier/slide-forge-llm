#!/usr/bin/env python3
"""Test script to verify the OpenEnv-RL (SlideForge) setup."""

import sys
import os

def main():
    print("=" * 60)
    print("OpenEnv-RL (SlideForge) Setup Test")
    print("=" * 60)
    
    # Step 1: Check Python version
    print(f"\nPython version: {sys.version}")
    
    # Step 2: Test core imports
    print("\n--- Testing core imports ---")
    
    try:
        import pydantic
        print("✓ pydantic")
    except ImportError as e:
        print(f"✗ pydantic: {e}")
        return False
    
    try:
        import httpx
        print("✓ httpx")
    except ImportError as e:
        print(f"✗ httpx: {e}")
        return False
    
    try:
        from PIL import Image
        print("✓ Pillow")
    except ImportError as e:
        print(f"✗ Pillow: {e}")
        return False
    
    try:
        import boto3
        print("✓ boto3")
    except ImportError as e:
        print(f"✗ boto3: {e}")
        return False
    
    try:
        import datasets
        print("✓ datasets")
    except ImportError as e:
        print(f"✗ datasets: {e}")
        return False
    
    try:
        import huggingface_hub
        print("✓ huggingface_hub")
    except ImportError as e:
        print(f"✗ huggingface_hub: {e}")
        return False
    
    # Step 3: Test SlideForge imports
    print("\n--- Testing SlideForge imports ---")
    
    try:
        from envs.slideforge_env.models import SlideBrief, SlideForgeAction, SlideForgeState
        print("✓ SlideForge models")
    except ImportError as e:
        print(f"✗ SlideForge models: {e}")
        return False
    
    try:
        from envs.slideforge_env.server.environment import SlideForgeEnvironment
        print("✓ SlideForgeEnvironment")
    except ImportError as e:
        print(f"✗ SlideForgeEnvironment: {e}")
        return False
    
    try:
        from training.rollouts import run_rollout, _create_bedrock_client
        print("✓ training.rollouts")
    except ImportError as e:
        print(f"✗ training.rollouts: {e}")
        return False
    
    try:
        from training.grpo_trainer import extract_tool_call
        print("✓ training.grpo_trainer")
    except ImportError as e:
        print(f"✗ training.grpo_trainer: {e}")
        return False
    
    try:
        from rewards.aggregator import compute_reward_details
        print("✓ rewards.aggregator")
    except ImportError as e:
        print(f"✗ rewards.aggregator: {e}")
        return False
    
    # Step 4: Test SlideForge environment
    print("\n--- Testing SlideForge environment ---")
    try:
        env = SlideForgeEnvironment()
        brief = {"topic": "Test Topic", "audience": "engineers", "num_slides": 3}
        obs = env.reset(brief=brief)
        print(f"✓ Environment reset")
        print(f"  Episode ID: {env.state.episode_id[:8]}...")
        print(f"  Phase: {obs.phase}")
        print(f"  Result: {obs.result[:50]}...")
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Test tool execution
    print("\n--- Testing tool execution ---")
    try:
        action = SlideForgeAction(
            tool="create_outline",
            parameters={"sections": ["Introduction", "Main Points", "Conclusion"]}
        )
        obs = env.step(action)
        print(f"✓ Tool execution: create_outline")
        print(f"  Success: {obs.success}")
        print(f"  Phase: {obs.phase}")
    except Exception as e:
        print(f"✗ Tool execution failed: {e}")
        return False
    
    # Step 6: Test AWS credentials
    print("\n--- Testing AWS credentials ---")
    try:
        session = boto3.Session()
        credentials = session.get_credentials()
        if credentials:
            print(f"✓ AWS credentials found")
            print(f"  Access Key: {credentials.access_key[:8]}..." if credentials.access_key else "  No access key")
            print(f"  Region: {session.region_name or 'not set (will use us-east-1)'}")
        else:
            print("⚠ No AWS credentials found - rollouts will fail")
            print("  Set up credentials in ~/.aws/credentials")
    except Exception as e:
        print(f"⚠ AWS credentials check failed: {e}")
    
    # Step 7: Test Bedrock client creation (without making API call)
    print("\n--- Testing Bedrock client creation ---")
    try:
        client = _create_bedrock_client()
        print(f"✓ Bedrock client created")
        print(f"  Region: {client.meta.region_name}")
    except Exception as e:
        print(f"✗ Bedrock client creation failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print("\nYou can now run rollouts with:")
    print("  python training/rollouts.py --num-rollouts 1 --max-turns 10")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
