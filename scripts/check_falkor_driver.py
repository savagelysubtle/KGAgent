"""Check FalkorDB driver signature."""
import inspect
from graphiti_core.driver.falkordb_driver import FalkorDriver

print("FalkorDriver.execute_query signature:")
print(inspect.signature(FalkorDriver.execute_query))

print("\nSource code:")
print(inspect.getsource(FalkorDriver.execute_query))

