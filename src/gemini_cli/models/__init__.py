"""
Model interfaces for different LLM providers.
"""

# Import models that should be available
from .gemini import GeminiModel

# Vertex AI support is conditional and loaded on demand
# Define a placeholder that will be replaced if the real implementation is available
VERTEX_SDK_AVAILABLE = False

class VertexAIModel:
    """Placeholder for VertexAIModel that will be replaced if available."""
    def __init__(self, *args, **kwargs):
        raise ImportError("Vertex AI SDK not available. Install with 'pip install google-cloud-aiplatform>=1.56.0'")

# Try to import the real implementation - this will replace the placeholder if successful
try:
    from .vertex import VertexAIModel, VERTEX_SDK_AVAILABLE
except (ImportError, ModuleNotFoundError, SyntaxError, AttributeError):
    # The import failed, but we already have the placeholder defined
    pass