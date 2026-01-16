from duckduckgo_search import DDGS
import json
import time

def test_search():
    print("Testing DDGS...")
    try:
        # Test 1: Simple query
        with DDGS() as ddgs:
            print("Query 1: 'test'")
            results = list(ddgs.text("test", max_results=3))
            print(f"Found {len(results)} results")
            for r in results:
                print(f"- {r.get('title')}")
            
            time.sleep(2)
            
            # Test 2: Complex query (simulating agent)
            q2 = "China Economy Crisis latest news analysis"
            print(f"\nQuery 2: '{q2}'")
            results2 = list(ddgs.text(q2, max_results=3))
            print(f"Found {len(results2)} results")
            for r in results2:
                 print(f"- {r.get('title')}")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")

if __name__ == "__main__":
    test_search()
