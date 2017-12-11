class MissingEmbedding(Exception):
    """No embedding saved for the specified tag"""


class UniqueEmbeddingTagError(Exception):
    """Raised when trying to overwrite an embedding without a unique tag."""
