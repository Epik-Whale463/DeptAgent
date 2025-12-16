"""
Advanced text splitter for optimal RAG performance
Uses recursive character splitting with semantic awareness
"""

from typing import List


class RecursiveCharacterSplitter:
    """
    Split text using recursive approach for better semantic coherence
    Follows LangChain's best practices
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100,
                 separators: List[str] = None):
        """
        Initialize text splitter
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to try in order
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Default separators maintain semantic structure better
        self.separators = separators or [
            "\n\n",      # Paragraph breaks (best)
            "\n",        # Line breaks
            ". ",        # Sentence breaks
            " ",         # Word breaks
            ""           # Character breaks (last resort)
        ]
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks with semantic awareness
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        return self._split_text_recursive(text, self.separators)
    
    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using provided separators
        
        Args:
            text: Text to split
            separators: Separators to try
            
        Returns:
            List of text chunks
        """
        final_chunks = []
        
        # Use the first separator that actually splits the text
        separator = separators[-1]
        for _s in separators:
            if _s == "":
                separator = _s
                break
            if _s in text:
                separator = _s
                break
        
        # Split text
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        
        # Merge small chunks back together
        good_splits = []
        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                # Chunk is too large, need to split further
                if good_splits:
                    merged_text = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged_text)
                    good_splits = []
                
                # Try next separator
                other_info = self._split_text_recursive(s, separators[separators.index(separator) + 1:])
                final_chunks.extend(other_info)
        
        if good_splits:
            merged_text = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged_text)
        
        return [chunk for chunk in final_chunks if chunk.strip()]  # Remove empty chunks
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """
        Merge splits together respecting chunk size limits
        
        Args:
            splits: List of text pieces to merge
            separator: Separator used between pieces
            
        Returns:
            List of merged chunks
        """
        separator_len = len(separator)
        
        good_splits = []
        current_chunk = []
        current_size = 0
        
        for split in splits:
            split_len = len(split)
            
            # Check if adding this split would exceed chunk_size
            if current_size + split_len + separator_len > self.chunk_size and current_chunk:
                # Save current chunk and start new one
                doc = separator.join(current_chunk)
                good_splits.append(doc)
                
                # Create overlap by including end of previous chunk
                current_chunk = self._get_overlap_splits(current_chunk, separator)
                current_size = sum(len(s) for s in current_chunk) + separator_len * (len(current_chunk) - 1)
            
            current_chunk.append(split)
            current_size += split_len + separator_len
        
        # Add final chunk
        if current_chunk:
            doc = separator.join(current_chunk)
            good_splits.append(doc)
        
        return good_splits
    
    def _get_overlap_splits(self, splits: List[str], separator: str) -> List[str]:
        """
        Get overlap chunks for next iteration
        
        Args:
            splits: Previous splits
            separator: Separator between splits
            
        Returns:
            Overlapping splits
        """
        if len(splits) == 0:
            return []
        
        # Keep building overlap until we reach chunk_overlap
        overlap_splits = []
        cumulative_len = 0
        
        for i in range(len(splits) - 1, -1, -1):
            split_len = len(splits[i])
            cumulative_len += split_len + len(separator)
            
            overlap_splits.insert(0, splits[i])
            
            if cumulative_len > self.chunk_overlap:
                break
        
        return overlap_splits
