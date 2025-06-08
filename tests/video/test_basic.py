def test_import():
    try:
        import spygate

        print("Successfully imported spygate")
    except ImportError as e:
        print(f"Failed to import spygate: {e}")
