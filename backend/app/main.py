from __future__ import annotations

import asyncio
import logging
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.agent.graph import BioAgent, extract_turn_artifacts, final_answer, has_tool_protocol_error
from app.config import get_settings
from app.rag.store import SUPPORTED_SUFFIXES, KnowledgeStore
from app.schemas import (
    Attachment,
    AuthConfig,
    AuthUser,
    ChatRequest,
    ChatResponse,
    ConversationDetail,
    ConversationSummary,
    HealthResponse,
    KnowledgeStatus,
    LoginRequest,
    RegisterRequest,
    Source,
    TrainingTriggerRequest,
)
from app.services.attachments import (
    SUPPORTED_ATTACHMENT_SUFFIXES,
    AttachmentNotFoundError,
    AttachmentStore,
    FileContentError,
    attachment_sources,
)
from app.services.auth import (
    AuthStore,
    DuplicateEmailError,
    InvalidCredentialsError,
    InvalidInviteError,
    RegistrationDisabledError,
)
from app.services.history import ConversationStore
from app.services.llm import is_loopback_base_url
from app.services.training import LocalSFTCoordinator, TrainingJobBusyError, TrainingJobState
from biocoder.bad_cases.schema import BadCase
from biocoder.bad_cases.store import BadCaseStore, should_create_bad_case
from biocoder.memory.store import SemanticMemoryStore
from biocoder.security.validation import ensure_path_within
from biocoder.state import AgentBudget, AgentState, AgentStatus, utc_now
from biocoder.trajectory.recorder import TrajectoryRecorder, use_recorder
from biocoder.trajectory.schema import ActionType, FailureType
from biocoder.trajectory.storage import TrajectoryStorage
from eval.benchmark import AgentPrediction, BenchmarkCase
from eval.evaluators.rule_evaluator import RuleEvaluator
from feedback.schema import FeedbackRecord, FeedbackRequest, FeedbackType
from feedback.store import FeedbackStore
from observability.logging import log_event
from observability.tracing import TaskContext, use_task_context

settings = get_settings()
knowledge_store = KnowledgeStore(settings)
attachment_store = AttachmentStore(settings)
conversation_store = ConversationStore(settings.history_db_path)
auth_store = AuthStore(settings.history_db_path)
trajectory_storage = TrajectoryStorage(settings.trajectory_dir, settings.trajectory_jsonl_path)
feedback_store = FeedbackStore(settings.feedback_dir)
training_coordinator = LocalSFTCoordinator(settings, feedback_store=feedback_store)
bad_case_store = BadCaseStore(settings.bad_case_dir)
semantic_memory_store = SemanticMemoryStore(
    settings.semantic_memory_dir,
    minimum_write_quality=settings.memory_minimum_quality,
)
_agent: BioAgent | None = None
logger = logging.getLogger("uvicorn.error.bioagent")


def get_agent() -> BioAgent:
    global _agent
    if _agent is None:
        _agent = BioAgent(settings, knowledge_store)
    return _agent


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=settings.auth_cookie_name,
        value=token,
        max_age=settings.auth_session_days * 24 * 60 * 60,
        httponly=True,
        secure=settings.app_env.lower() == "production",
        samesite="lax",
        path="/",
    )


def require_user(request: Request) -> AuthUser:
    token = request.cookies.get(settings.auth_cookie_name, "")
    user = auth_store.user_for_session(token) if token else None
    if user is None:
        raise HTTPException(status_code=401, detail="请先登录后继续。")
    return user


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    attachment_store.initialize()
    auth_store.initialize()
    conversation_store.initialize()
    yield


app = FastAPI(
    title="BioCoder 2.0 API",
    version="2.0.0",
    description="Self-improving LLM + RAG + Tool Calling biomedical research agent",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/auth/config", response_model=AuthConfig)
def auth_config() -> AuthConfig:
    return AuthConfig(
        registration_enabled=settings.auth_registration_enabled,
        invite_required=bool(settings.auth_invite_code),
        admin_email=settings.auth_admin_email,
    )


@app.post("/api/auth/register", response_model=AuthUser, status_code=201)
def register(payload: RegisterRequest, response: Response) -> AuthUser:
    try:
        user = auth_store.register(
            email=payload.email,
            display_name=payload.display_name,
            password=payload.password,
            invite_code=payload.invite_code,
            expected_invite_code=settings.auth_invite_code,
            registration_enabled=settings.auth_registration_enabled,
        )
    except (InvalidInviteError, RegistrationDisabledError) as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except DuplicateEmailError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    token = auth_store.create_session(user.id, duration_days=settings.auth_session_days)
    _set_session_cookie(response, token)
    return user


@app.post("/api/auth/login", response_model=AuthUser)
def login(payload: LoginRequest, response: Response) -> AuthUser:
    try:
        user = auth_store.authenticate(email=payload.email, password=payload.password)
    except InvalidCredentialsError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    token = auth_store.create_session(user.id, duration_days=settings.auth_session_days)
    _set_session_cookie(response, token)
    return user


@app.get("/api/auth/me", response_model=AuthUser)
def me(user: AuthUser = Depends(require_user)) -> AuthUser:
    return user


@app.post("/api/auth/logout", status_code=204)
def logout(request: Request, response: Response) -> Response:
    token = request.cookies.get(settings.auth_cookie_name, "")
    if token:
        auth_store.revoke_session(token)
    response.delete_cookie(
        key=settings.auth_cookie_name,
        secure=settings.app_env.lower() == "production",
        samesite="lax",
        path="/",
    )
    response.status_code = 204
    return response


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        llm_configured=(
            bool(settings.vllm_base_url)
            if settings.model_provider.lower() == "vllm"
            else bool(settings.openai_api_key)
            or is_loopback_base_url(settings.openai_base_url)
        ),
        knowledge_ready=knowledge_store.ready,
        attachments_enabled=settings.attachments_enabled,
        vision_input_enabled=settings.vision_input_enabled,
        attachment_formats=sorted(SUPPORTED_ATTACHMENT_SUFFIXES),
        attachment_max_files=settings.attachment_max_files,
        attachment_max_file_bytes=settings.attachment_max_file_bytes,
    )


@app.post("/api/attachments", response_model=Attachment, status_code=201)
async def upload_attachment(
    file: UploadFile = File(...), user: AuthUser = Depends(require_user)
) -> Attachment:
    if not settings.attachments_enabled:
        raise HTTPException(status_code=403, detail="附件功能未启用。")
    content = await file.read(settings.attachment_max_file_bytes + 1)
    await file.close()
    if len(content) > settings.attachment_max_file_bytes:
        maximum = settings.attachment_max_file_bytes // (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"单个附件不能超过 {maximum} MB。")
    try:
        attachment = await asyncio.to_thread(attachment_store.save, file.filename, content)
        auth_store.claim_resource("attachment", attachment.id, user.id)
        return attachment
    except FileContentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user: AuthUser = Depends(require_user)) -> ChatResponse:
    if any(
        not auth_store.owns_resource("attachment", attachment_id, user.id)
        for attachment_id in request.attachment_ids
    ):
        raise HTTPException(status_code=404, detail="附件不存在或无权访问。")
    if request.thread_id and not conversation_store.owns_thread(request.thread_id, user.id):
        raise HTTPException(status_code=404, detail="会话不存在或无权访问。")
    try:
        prepared_attachments = await asyncio.to_thread(
            attachment_store.prepare, request.attachment_ids
        )
    except AttachmentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileContentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    thread_id = request.thread_id or str(uuid.uuid4())
    context = TaskContext.create(thread_id)
    state = AgentState(
        task_id=context.task_id,
        trace_id=context.trace_id,
        session_id=context.session_id,
        user_query=request.message,
        status=AgentStatus.RUNNING,
        model_version=settings.model_version or settings.vllm_model or settings.llm_model,
        prompt_version=settings.prompt_version,
        agent_version=settings.agent_version,
        budget=AgentBudget(
            max_steps=settings.max_steps,
            max_retries=settings.max_retries,
            timeout_seconds=settings.agent_timeout_seconds,
            token_budget=settings.token_budget,
            cost_budget=settings.cost_budget,
        ),
    )
    recorder = TrajectoryRecorder(state)
    with use_task_context(context), use_recorder(recorder):
        log_event(
            logger,
            "request",
            query_length=len(request.message),
            attachment_count=len(prepared_attachments),
        )
        history = conversation_store.messages_for_agent(thread_id, user.id)
        memory_records = semantic_memory_store.search(
            request.message, limit=3, owner_id=user.id
        )
        memory_context = [record.content for record in memory_records]
        recorder.state.memory_context = [record.model_dump(mode="json") for record in memory_records]
        recorder.record(
            ActionType.REQUEST,
            name="chat_request",
            arguments={
                "query": request.message,
                "attachments": [item.descriptor.name for item in prepared_attachments],
            },
            observation={
                "history_messages": len(history),
                "attachment_count": len(prepared_attachments),
            },
        )
        recorder.record(
            ActionType.MEMORY,
            name="memory_retrieval",
            arguments={"limit": 3},
            observation={
                "working_memory_messages": len(history),
                "semantic_memory_ids": [record.memory_id for record in memory_records],
            },
        )
        try:
            result = await asyncio.wait_for(
                get_agent().ask(
                    request.message,
                    history,
                    memory_context,
                    attachments=prepared_attachments,
                ),
                timeout=settings.agent_timeout_seconds,
            )
        except TimeoutError as exc:
            log_event(logger, "request_failed", level=logging.WARNING, failure_type="TIMEOUT")
            failed = recorder.fail(str(exc), FailureType.TIMEOUT)
            trajectory_storage.save(failed)
            bad_case_store.add(
                BadCase(
                    task_id=failed.task_id,
                    query=failed.query,
                    trajectory=failed.model_dump(mode="json"),
                    answer="",
                    failure_type=FailureType.TIMEOUT,
                    model_version=failed.model_version,
                    provenance={"trigger": "task_failed"},
                )
            )
            raise HTTPException(
                status_code=504,
                detail=f"Agent execution exceeded {settings.agent_timeout_seconds:.0f} seconds. Check the model endpoint or try again.",
            ) from exc
        except RuntimeError as exc:
            log_event(
                logger,
                "request_failed",
                level=logging.WARNING,
                failure_type="UNKNOWN",
                error=str(exc),
            )
            failed = recorder.fail(str(exc), FailureType.UNKNOWN)
            trajectory_storage.save(failed)
            bad_case_store.add(
                BadCase(
                    task_id=failed.task_id,
                    query=failed.query,
                    trajectory=failed.model_dump(mode="json"),
                    failure_type=FailureType.UNKNOWN,
                    model_version=failed.model_version,
                    provenance={"trigger": "task_failed"},
                )
            )
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            log_event(
                logger,
                "request_failed",
                level=logging.ERROR,
                failure_type="UNKNOWN",
                error=str(exc),
            )
            failed = recorder.fail(str(exc), FailureType.UNKNOWN)
            trajectory_storage.save(failed)
            bad_case_store.add(
                BadCase(
                    task_id=failed.task_id,
                    query=failed.query,
                    trajectory=failed.model_dump(mode="json"),
                    failure_type=FailureType.UNKNOWN,
                    model_version=failed.model_version,
                    provenance={"trigger": "task_failed"},
                )
            )
            raise HTTPException(status_code=502, detail=f"Agent execution failed: {exc}") from exc

        messages = result.get("messages", [])
        retrieved_sources, tools_used = extract_turn_artifacts(messages)
        direct_sources = attachment_sources(prepared_attachments)
        source_keys = {(source.title, source.url) for source in retrieved_sources}
        sources = [
            *retrieved_sources,
            *[
                source
                for source in direct_sources
                if (source.title, source.url) not in source_keys
            ],
        ]
        if prepared_attachments:
            tools_used = ["read_attachment", *tools_used]
        answer = final_answer(messages)
        tool_protocol_error = has_tool_protocol_error(messages)
        response = ChatResponse(
            thread_id=thread_id,
            task_id=context.task_id,
            trace_id=context.trace_id,
            answer=answer,
            plan=result.get("plan", []),
            sources=sources,
            tools_used=tools_used,
            attachments=[item.descriptor for item in prepared_attachments],
        )
        auth_store.claim_resource("task", context.task_id, user.id)
        recorder.state.plan = response.plan
        recorder.state.artifacts = [source.model_dump(mode="json") for source in sources]
        tool_steps = [
            step for step in recorder.trajectory.steps if step.action.type == ActionType.TOOL_CALL
        ]
        online_case = BenchmarkCase(
            id=context.task_id,
            task_type=state.task_type,
            query=request.message,
        )
        online_prediction = AgentPrediction(
            id=context.task_id,
            answer=answer,
            plan=response.plan,
            tools_used=tools_used,
            tool_calls=[
                {
                    "name": step.action.tool,
                    "arguments": step.action.arguments,
                }
                for step in tool_steps
            ],
            sources=[source.model_dump(mode="json") for source in sources],
            success=state.status != AgentStatus.BUDGET_EXCEEDED and not tool_protocol_error,
            errors=[
                {"source": "tool", "message": step.error}
                for step in tool_steps
                if step.error
            ]
            + (
                [
                    {
                        "source": "model",
                        "type": "TOOL_PROTOCOL_ERROR",
                        "message": "Model emitted internal tool protocol instead of a final answer.",
                    }
                ]
                if tool_protocol_error
                else []
            ),
            step_count=len(recorder.trajectory.steps),
            latency_ms=(utc_now() - recorder.state.start_time).total_seconds() * 1000,
            prompt_tokens=recorder.state.token_usage.prompt_tokens,
            completion_tokens=recorder.state.token_usage.completion_tokens,
            estimated_cost=recorder.state.estimated_cost,
            loop_detected=state.status == AgentStatus.BUDGET_EXCEEDED,
        )
        evaluation = RuleEvaluator().evaluate(online_case, online_prediction)
        recorder.record(
            ActionType.EVALUATION,
            name="online_rule_evaluator",
            observation=evaluation.model_dump(mode="json"),
        )
        if evaluation.task_success and evaluation.score >= settings.memory_minimum_quality:
            memory_record, created = semantic_memory_store.add(
                query=request.message,
                content=answer,
                quality_score=evaluation.score,
                source_task=context.task_id,
                model_version=state.model_version,
                owner_id=user.id,
                metadata={"tools_used": tools_used, "source_count": len(sources)},
            )
            recorder.record(
                ActionType.MEMORY,
                name="semantic_memory_write",
                observation={
                    "created": created,
                    "memory_id": memory_record.memory_id if memory_record else None,
                    "quality_score": evaluation.score,
                },
            )
        trajectory = recorder.finalize(
            answer,
            success=evaluation.task_success,
            metrics={
                **evaluation.model_dump(mode="json"),
                "tool_count": len(tools_used),
                "source_count": len(sources),
                "step_count": len(recorder.trajectory.steps) + 1,
            },
        )
        trajectory.failure_type = evaluation.failure_type
        trajectory_storage.save(trajectory)
        tool_failure = any(step.error for step in tool_steps)
        if should_create_bad_case(
            task_success=evaluation.task_success,
            score=evaluation.score,
            threshold=settings.bad_case_score_threshold,
            tool_failure=tool_failure,
        ):
            bad_case_store.add(
                BadCase(
                    task_id=trajectory.task_id,
                    query=trajectory.query,
                    trajectory=trajectory.model_dump(mode="json"),
                    answer=answer,
                    score=evaluation.score,
                    failure_type=evaluation.failure_type or FailureType.UNKNOWN,
                    model_version=trajectory.model_version,
                    provenance={"trigger": "online_evaluation"},
                )
            )
        log_event(
            logger,
            "request_completed",
            tools=tools_used,
            sources=len(sources),
            latency_ms=recorder.state.latency_ms,
            token_usage=recorder.state.token_usage.model_dump(),
            estimated_cost=recorder.state.estimated_cost,
        )
        try:
            conversation_store.save_turn(
                thread_id, request.message, response, user_id=user.id
            )
        except Exception as exc:
            log_event(logger, "history_persist_failed", level=logging.ERROR, error=str(exc))
        return response


@app.post("/api/feedback", response_model=FeedbackRecord)
def submit_feedback(
    request: FeedbackRequest, user: AuthUser = Depends(require_user)
) -> FeedbackRecord:
    if not auth_store.owns_resource("task", request.task_id, user.id):
        raise HTTPException(status_code=404, detail="回答记录不存在或无权访问。")
    trajectory = trajectory_storage.load(request.task_id)
    if trajectory is None:
        raise HTTPException(status_code=404, detail="Task trajectory not found")
    if request.feedback_type in {
        FeedbackType.THUMBS_UP,
        FeedbackType.THUMBS_DOWN,
        FeedbackType.RATING,
    } and feedback_store.has_score(request.task_id):
        raise HTTPException(status_code=409, detail="A score has already been submitted for this answer")
    record = feedback_store.add(request)
    if record.is_negative:
        semantic_memory_store.deactivate_by_source_task(request.task_id, owner_id=user.id)
    if record.is_negative or record.corrected_answer:
        score = float(trajectory.metrics.get("score", 0))
        bad_case_store.add(
            BadCase(
                task_id=trajectory.task_id,
                query=trajectory.query,
                trajectory=trajectory.model_dump(mode="json"),
                answer=trajectory.final_answer or "",
                score=score,
                failure_type=trajectory.failure_type or FailureType.UNKNOWN,
                feedback=record.text_feedback or record.corrected_answer or record.feedback_type.value,
                model_version=trajectory.model_version,
                provenance={"trigger": "human_feedback", "feedback_id": record.feedback_id},
            )
        )
    training_coordinator.maybe_schedule(record)
    return record


def _require_local_training_api(request: Request) -> None:
    if not settings.training_api_enabled:
        raise HTTPException(status_code=403, detail="Local training API is disabled")
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        raise HTTPException(status_code=403, detail="Local training API only accepts loopback clients")


@app.get("/api/training/status", response_model=TrainingJobState)
def training_status(request: Request) -> TrainingJobState:
    _require_local_training_api(request)
    return training_coordinator.snapshot()


@app.post("/api/training/sft", response_model=TrainingJobState, status_code=202)
def trigger_sft(request: Request, payload: TrainingTriggerRequest) -> TrainingJobState:
    _require_local_training_api(request)
    if payload.execute and not settings.training_api_allow_execute:
        raise HTTPException(status_code=403, detail="Executable local training is disabled")
    try:
        return training_coordinator.schedule(execute=payload.execute, trigger="manual_api")
    except TrainingJobBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/conversations", response_model=list[ConversationSummary])
def list_conversations(
    limit: int = Query(default=50, ge=1, le=200),
    user: AuthUser = Depends(require_user),
) -> list[ConversationSummary]:
    return [
        ConversationSummary.model_validate(row)
        for row in conversation_store.list_conversations(user.id, limit)
    ]


@app.get("/api/conversations/{thread_id}", response_model=ConversationDetail)
def get_conversation(
    thread_id: str, user: AuthUser = Depends(require_user)
) -> ConversationDetail:
    conversation = conversation_store.get_conversation(thread_id, user.id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationDetail.model_validate(conversation)


@app.delete("/api/conversations/{thread_id}", status_code=204)
def delete_conversation(
    thread_id: str, user: AuthUser = Depends(require_user)
) -> Response:
    if not conversation_store.delete_conversation(thread_id, user.id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return Response(status_code=204)


@app.get("/api/knowledge", response_model=KnowledgeStatus)
def knowledge_status(_: AuthUser = Depends(require_user)) -> KnowledgeStatus:
    return KnowledgeStatus.model_validate(knowledge_store.status())


@app.get("/api/knowledge/search", response_model=list[Source])
def search_knowledge(
    query: str = Query(min_length=1, max_length=1000),
    top_k: int = Query(default=6, ge=1, le=10),
    _: AuthUser = Depends(require_user),
) -> list[Source]:
    try:
        return [Source.model_validate(row) for row in knowledge_store.search(query, top_k)]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Knowledge search failed: {exc}") from exc


@app.post("/api/knowledge/reindex", response_model=KnowledgeStatus)
def reindex_knowledge(_: AuthUser = Depends(require_user)) -> KnowledgeStatus:
    try:
        return KnowledgeStatus.model_validate(knowledge_store.rebuild())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc


@app.post("/api/knowledge/upload", response_model=KnowledgeStatus)
async def upload_knowledge(
    file: UploadFile = File(...), _: AuthUser = Depends(require_user)
) -> KnowledgeStatus:
    original = Path(file.filename or "document.txt")
    suffix = original.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Supported formats: .md, .txt, .pdf, .docx, .json",
        )
    safe_stem = re.sub(r"[^a-zA-Z0-9._\-\u4e00-\u9fff]", "_", original.stem)[:80] or "document"
    destination = settings.uploads_dir / f"{safe_stem}{suffix}"
    try:
        destination = ensure_path_within(destination, settings.uploads_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid upload path") from exc
    content = await file.read()
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File size must not exceed 15 MB")
    destination.write_bytes(content)
    try:
        return KnowledgeStatus.model_validate(knowledge_store.rebuild())
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Upload succeeded but indexing failed: {exc}") from exc
