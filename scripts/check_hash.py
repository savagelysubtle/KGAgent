import hashlib
import sys
sys.stdout.reconfigure(line_buffering=True)

# Check what different filenames hash to
filenames = ['chat', 'chat.html', '31e06f7d89feb99a']

for name in filenames:
    h = hashlib.sha256(name.encode()).hexdigest()[:16]
    print(f"'{name}' -> '{h}'")

# Also check if 31e06f7d89feb99a is a hash of any common name
print("\nNote: 31e06f7d89feb99a is the hashed filename in the raw directory")

