"""Entity resolution and deduplication service."""

import re
from typing import List, Dict, Set, Tuple
from collections import defaultdict

from ..models.entity import Entity, Relationship
from ..core.logging import logger


class EntityResolver:
    """
    Resolves and merges duplicate entities across chunks and documents.
    Uses fuzzy matching and alias resolution.
    """

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize the entity resolver.

        Args:
            similarity_threshold: Minimum similarity for entity matching (0-1)
        """
        self.similarity_threshold = similarity_threshold

    def normalize_name(self, name: str) -> str:
        """
        Normalize entity name for comparison.

        Args:
            name: Entity name to normalize

        Returns:
            Normalized name (lowercase, stripped, no extra spaces)
        """
        # Lowercase, remove extra spaces, strip punctuation
        normalized = name.lower().strip()
        normalized = re.sub(r"[^\w\s]", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    def names_match(self, name1: str, name2: str) -> bool:
        """
        Check if two entity names refer to the same entity.

        Args:
            name1: First entity name
            name2: Second entity name

        Returns:
            True if names likely refer to same entity
        """
        n1 = self.normalize_name(name1)
        n2 = self.normalize_name(name2)

        # Exact match
        if n1 == n2:
            return True

        # Empty check
        if not n1 or not n2:
            return False

        # One is substring of other (handles "OpenAI" vs "OpenAI Inc")
        if n1 in n2 or n2 in n1:
            return True

        # Check word overlap for multi-word names
        words1 = set(n1.split())
        words2 = set(n2.split())

        if len(words1) > 1 and len(words2) > 1:
            # Jaccard similarity on words
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            if union > 0:
                overlap = intersection / union
                if overlap >= self.similarity_threshold:
                    return True

        # Check if significant words match (ignoring common words)
        common_words = {"the", "a", "an", "of", "in", "for", "and", "or", "inc", "corp", "llc", "ltd"}
        significant1 = words1 - common_words
        significant2 = words2 - common_words

        if significant1 and significant2:
            if significant1 == significant2:
                return True
            # Check if one is subset of other
            if significant1.issubset(significant2) or significant2.issubset(significant1):
                return True

        return False

    def resolve_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Merge duplicate entities into canonical entities.

        Args:
            entities: List of potentially duplicate entities

        Returns:
            List of deduplicated, merged entities
        """
        if not entities:
            return []

        logger.info(f"Resolving {len(entities)} entities...")

        # Group entities by type first, then by normalized name
        type_groups: Dict[str, List[Entity]] = defaultdict(list)
        for entity in entities:
            type_groups[entity.type.value].append(entity)

        # Within each type, group by matching names
        merged_entities = []

        for entity_type, type_entities in type_groups.items():
            # Build groups of matching entities
            entity_groups: List[List[Entity]] = []

            for entity in type_entities:
                matched = False

                # Check against existing groups
                for group in entity_groups:
                    # Check if this entity matches any entity in the group
                    for existing in group:
                        if self.names_match(entity.name, existing.name):
                            group.append(entity)
                            matched = True
                            break
                        # Also check aliases
                        for alias in existing.aliases:
                            if self.names_match(entity.name, alias):
                                group.append(entity)
                                matched = True
                                break
                        if matched:
                            break
                    if matched:
                        break

                if not matched:
                    # Start a new group
                    entity_groups.append([entity])

            # Merge each group into a single canonical entity
            for group in entity_groups:
                merged = self._merge_entity_group(group)
                merged_entities.append(merged)

        logger.info(
            f"Resolved {len(entities)} entities into {len(merged_entities)} unique entities"
        )
        return merged_entities

    def _merge_entity_group(self, entities: List[Entity]) -> Entity:
        """
        Merge a group of duplicate entities into one canonical entity.

        Args:
            entities: List of entities to merge

        Returns:
            Single merged entity
        """
        if len(entities) == 1:
            return entities[0]

        # Use the most common/longest name as canonical
        name_counts: Dict[str, int] = defaultdict(int)
        for e in entities:
            name_counts[e.name] += 1

        # Prefer longer names with higher counts
        canonical_name = max(
            name_counts.keys(), key=lambda n: (name_counts[n], len(n))
        )

        # Merge all aliases
        all_aliases: Set[str] = set()
        for e in entities:
            all_aliases.update(e.aliases)
            if e.name != canonical_name:
                all_aliases.add(e.name)
        all_aliases.discard(canonical_name)

        # Merge source chunks and documents
        all_chunks: Set[str] = set()
        all_docs: Set[str] = set()
        for e in entities:
            all_chunks.update(e.source_chunks)
            all_docs.update(e.source_documents)

        # Use the longest/most detailed description
        descriptions = [e.description for e in entities if e.description]
        best_description = (
            max(descriptions, key=lambda d: len(d)) if descriptions else None
        )

        # Average confidence
        avg_confidence = sum(e.confidence for e in entities) / len(entities)

        # Merge metadata
        merged_metadata: Dict = {}
        for e in entities:
            merged_metadata.update(e.metadata)

        return Entity(
            name=canonical_name,
            type=entities[0].type,
            description=best_description,
            aliases=list(all_aliases),
            source_chunks=list(all_chunks),
            source_documents=list(all_docs),
            confidence=avg_confidence,
            metadata=merged_metadata,
        )

    def resolve_relationships(
        self, relationships: List[Relationship], resolved_entities: List[Entity]
    ) -> List[Relationship]:
        """
        Update relationships to use canonical entity names and deduplicate.

        Args:
            relationships: Original relationships
            resolved_entities: Deduplicated entities with aliases

        Returns:
            Relationships with canonical entity names, deduplicated
        """
        if not relationships or not resolved_entities:
            return []

        logger.info(f"Resolving {len(relationships)} relationships...")

        # Build name -> canonical name mapping
        name_map: Dict[str, str] = {}
        entity_names: Set[str] = set()

        for entity in resolved_entities:
            canonical = entity.name
            entity_names.add(canonical)
            name_map[self.normalize_name(canonical)] = canonical
            for alias in entity.aliases:
                name_map[self.normalize_name(alias)] = canonical

        resolved_rels: List[Relationship] = []
        seen_rels: Set[Tuple[str, str, str]] = set()

        for rel in relationships:
            # Resolve source and target to canonical names
            source_norm = self.normalize_name(rel.source_entity)
            target_norm = self.normalize_name(rel.target_entity)

            source_canonical = name_map.get(source_norm)
            target_canonical = name_map.get(target_norm)

            # Try fuzzy matching if exact match fails
            if not source_canonical:
                for entity in resolved_entities:
                    if self.names_match(rel.source_entity, entity.name):
                        source_canonical = entity.name
                        break

            if not target_canonical:
                for entity in resolved_entities:
                    if self.names_match(rel.target_entity, entity.name):
                        target_canonical = entity.name
                        break

            # Skip if we can't resolve both entities
            if not source_canonical or source_canonical not in entity_names:
                logger.debug(f"Could not resolve source entity: {rel.source_entity}")
                continue
            if not target_canonical or target_canonical not in entity_names:
                logger.debug(f"Could not resolve target entity: {rel.target_entity}")
                continue

            # Skip self-references
            if source_canonical == target_canonical:
                continue

            # Deduplicate relationships (same source, target, type)
            rel_key = (source_canonical, target_canonical, rel.type.value)
            if rel_key in seen_rels:
                continue
            seen_rels.add(rel_key)

            resolved_rels.append(
                Relationship(
                    source_entity=source_canonical,
                    target_entity=target_canonical,
                    type=rel.type,
                    description=rel.description,
                    source_chunk=rel.source_chunk,
                )
            )

        logger.info(
            f"Resolved {len(relationships)} relationships into {len(resolved_rels)} unique relationships"
        )
        return resolved_rels

