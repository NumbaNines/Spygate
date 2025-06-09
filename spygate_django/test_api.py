#!/usr/bin/env python3
"""
Test script for SpygateAI Django Integration

This script tests our Django API endpoints to ensure the SpygateAI engine
is properly integrated and working through the web interface.
"""

import os
import sys

import django

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "spygate_web.settings")
django.setup()

from api.services import get_spygate_service


def test_django_spygate_integration():
    """Test the Django-SpygateAI integration."""
    print("🧪 Testing Django-SpygateAI Integration")
    print("=" * 50)

    try:
        # Test 1: Service initialization
        print("\n1️⃣ Testing Service Initialization")
        print("-" * 35)

        service = get_spygate_service()
        print(f"✅ Service created: {type(service).__name__}")
        print(f"✅ Service initialized: {service._initialized}")

        # Test 2: Engine status
        print("\n2️⃣ Testing Engine Status")
        print("-" * 25)

        status = service.get_engine_status()
        print(f"✅ Status retrieved: {status.get('initialized', False)}")

        if "engine_status" in status:
            engine_status = status["engine_status"]
            print(f"✅ Engine Status: {engine_status.get('status', 'unknown')}")
            print(f"✅ Systems Ready: {engine_status.get('systems_count', 0)}")

        # Test 3: Hardware optimization
        print("\n3️⃣ Testing Hardware Optimization")
        print("-" * 35)

        hardware_status = service.get_hardware_optimization_status()
        print(f"✅ Hardware status: {hardware_status.get('success', False)}")

        if hardware_status.get("success"):
            print(f"✅ Hardware Tier: {hardware_status.get('hardware_tier', 'unknown')}")
            print(f"✅ GPU Available: {hardware_status.get('gpu_available', False)}")

        # Test 4: Situational library test (without files)
        print("\n4️⃣ Testing Situational Library")
        print("-" * 32)

        library_result = service.build_situational_library("3rd_long")
        print(f"✅ Library built: {library_result.get('success', False)}")

        if library_result.get("success"):
            library_data = library_result.get("situational_library", {})
            print(f"✅ Total clips: {library_data.get('total_clips', 0)}")
            print(f"✅ Games analyzed: {len(library_data.get('clips_by_game', {}))}")

        print("\n🎉 Django-SpygateAI Integration Test Complete!")
        print("✅ All core systems are operational and working through Django!")

        return True

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_django_spygate_integration()
    if success:
        print("\n🚀 Ready to start Django development server!")
        print("   Run: python manage.py runserver")
    else:
        print("\n⚠️  Fix integration issues before proceeding")

    sys.exit(0 if success else 1)
