
from langchain_mcp_adapters.client import MultiServerMCPClient
import inspect

print("Methods:", [m for m in dir(MultiServerMCPClient) if not m.startswith('_')])
try:
    print("Init Signature:", inspect.signature(MultiServerMCPClient.__init__))
except Exception as e:
    print("Init Sig Error:", e)

# Check if there is a 'connect' method?
if hasattr(MultiServerMCPClient, 'connect_to_server'):
    print("Has connect_to_server")
else:
    print("No connect_to_server")
