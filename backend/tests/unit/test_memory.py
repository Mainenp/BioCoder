from biocoder.memory.store import SemanticMemoryStore


def test_semantic_memory_is_quality_gated_retrievable_and_decayable(tmp_path) -> None:
    store = SemanticMemoryStore(tmp_path / "memory", minimum_write_quality=0.75)

    rejected, created = store.add(
        query="EGFR low quality",
        content="unverified",
        quality_score=0.2,
        source_task="bad-task",
        model_version="model-a",
    )
    assert rejected is None
    assert created is False

    record, created = store.add(
        query="EGFR C797S resistance",
        content="C797S impairs covalent binding.",
        quality_score=0.9,
        source_task="good-task",
        model_version="model-a",
    )
    assert record is not None
    assert created is True
    assert store.search("C797S covalent inhibitor")[0].memory_id == record.memory_id

    updated = store.record_outcome(record.memory_id, success=False)
    assert updated.failure_count == 1
    assert updated.quality_score < 0.9
    assert store.decay(factor=0.5, deactivate_below=0.5) == 1
    assert store.get(record.memory_id).active is False  # type: ignore[union-attr]


def test_semantic_memory_is_isolated_by_owner(tmp_path) -> None:
    store = SemanticMemoryStore(tmp_path / "memory", minimum_write_quality=0.75)
    record, created = store.add(
        query="private target analysis",
        content="private research conclusion",
        quality_score=1.0,
        source_task="task-a",
        model_version="model-a",
        owner_id="user-a",
    )
    assert record is not None and created is True
    assert store.search("private target", owner_id="user-a")
    assert store.search("private target", owner_id="user-b") == []
    assert store.search("private target") == []


def test_semantic_memory_rejects_protocol_artifacts_and_deactivates_negative_tasks(
    tmp_path,
) -> None:
    store = SemanticMemoryStore(tmp_path / "memory", minimum_write_quality=0.75)
    rejected, created = store.add(
        query="AI drug pipeline",
        content="<｜｜DSML｜｜tool_calls></｜｜DSML｜｜tool_calls>",
        quality_score=1.0,
        source_task="protocol-task",
        model_version="model-a",
    )
    assert rejected is None
    assert created is False

    record, created = store.add(
        query="EGFR",
        content="Evidence-based answer",
        quality_score=1.0,
        source_task="rejected-task",
        model_version="model-a",
    )
    assert record is not None and created is True
    assert store.deactivate_by_source_task("rejected-task") == 1
    assert store.get(record.memory_id).active is False  # type: ignore[union-attr]
