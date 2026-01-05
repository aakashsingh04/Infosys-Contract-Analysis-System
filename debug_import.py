import sys
import os

print("Python Executable:", sys.executable)
print("Sys Path:")
for p in sys.path:
    print(p)

try:
    import pinecone
    print("\nPinecone imported successfully")
    print("Pinecone file:", getattr(pinecone, '__file__', 'No __file__'))
    print("Pinecone path:", getattr(pinecone, '__path__', 'No __path__'))
    print("Pinecone dir:", dir(pinecone))
except ImportError as e:
    print(f"\nImportError: {e}")

try:
    from pinecone import Pinecone
    print("\nPinecone class imported successfully")
except ImportError as e:
    print(f"\nFailed to import Pinecone class: {e}")
